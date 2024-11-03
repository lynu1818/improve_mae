# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .vit_mae_util.pos_embed import get_2d_sincos_pos_embed
from .hiera_utils import pretrained_model
from .hfhub import has_config


class DecoderUpsampleBlock(nn.Module):
    def __init__(self, D, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(D, D * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        # remove cls token
        cls_token = x[:, :1, :]
        x_ = x[:, 1:, :]

        B, N, D = x_.shape
        H = W = int(N ** .5)
        x_ = x_.view(B, H, W, D).permute(0, 3, 1, 2) # B, D, H, W
        # Upsample
        x_ = self.conv(x_) # B, D * (upscale_factor ** 2), H, W
        x_ = self.pixel_shuffle(x_) # B, D, H * upscale_factor, W * upscale_factor

        # Flatten
        x_ = x_.permute(0, 2, 3, 1).reshape(B, -1, D)
        # Concat cls token
        x = torch.cat([cls_token, x_], dim=1)
        return x


class EfficientMaskedAutoencoderPlusViT(nn.Module):
    """ Efficient Masked Autoencoder with VisionTransformer backbone
    EMAE: https://arxiv.org/abs/2302.14431
    """
    @has_config
    def __init__(self, model_name = '', input_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True,
                 drop_path_rate: float = 0.0):
        super().__init__()

        self.model_name = model_name
        img_size = input_size[0]
        self.input_size = input_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        if img_size == 224:
            self.decoder_upsample_layers = nn.ModuleDict({
                str(decoder_depth - 2): DecoderUpsampleBlock(decoder_embed_dim, 2),
            })
        elif img_size == 448:
            self.decoder_upsample_layers = nn.ModuleDict({
                str(decoder_depth - 2): DecoderUpsampleBlock(decoder_embed_dim, 2),
                str(decoder_depth - 1): DecoderUpsampleBlock(decoder_embed_dim, 2)

            })
        else:
            raise NotImplementedError(f"Unsupported input size: {img_size}")

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        #torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking with multiple strategies.
        Shuffles each sample individually to select visible patches and generate binary masks.
        
        Args:
            x (torch.Tensor): Input sequence of shape [N, L, D], where:
                N = batch size
                L = sequence length (number of patches)
                D = feature dimension per patch
            mask_ratio (float): Fraction of patches to mask for each strategy.

        Returns:
            torch.Tensor: Visible patches for each strategy, shape [N, num_mask_strategy, L//num_mask_strategy, D].
            torch.Tensor: Binary masks for each strategy, shape [N, num_mask_strategy, L], where 0 is keep and 1 is masked.
            torch.Tensor: Restore indices to reorder patches back to original order, shape [N, L].
        """
        num_mask_strategy = int(1 / (1 - mask_ratio))
        
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)


        x_visible_patches = torch.zeros((N, num_mask_strategy, len_keep, D), device=x.device)
        mask_strategies = torch.ones((N, num_mask_strategy, L), device=x.device)
        

        # divide the whole data for different masking strategy
        ids_keep_list = []
        for i in range(num_mask_strategy):
            # Calculate start and end indices for the i-th subset
            start_idx = i * (L // num_mask_strategy)
            end_idx = (i + 1) * (L // num_mask_strategy)
            
            # Obtain the i-th visible patches
            ids_keep_i = ids_shuffle[:, start_idx:end_idx]
            ids_keep_list.append(ids_keep_i)
            x_visible_patches[:, i] = torch.gather(x, dim=1, index=ids_keep_i.unsqueeze(-1).repeat(1, 1, D))

            # Obtain the i-th mask
            m_i = torch.ones([N, L], device=x.device)
            m_i[:, start_idx:end_idx] = 0
            # Unshuffle to get the binary mask
            mask_strategies[:, i] = torch.gather(m_i, dim=1, index=ids_restore)
        
        ids_keep_list = torch.stack(ids_keep_list, dim=1)
        return x_visible_patches, mask_strategies, ids_keep_list

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_keep_list = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], x.shape[1], -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        N, K, L, D = x.shape # N: batch size, K: num mask strategies, L: sequence length
        x = x.view(N * K, L, D)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x.view(N, K, L, D)

        return x, mask, ids_keep_list

    def forward_decoder(self, x, ids_keep_list):
        """
        Args:
            x (torch.Tensor): Input tensor after encoding, shape [B, K, L+1, D], where:
                B = batch size
                K = number of masking strategies
                L = number of patches per strategy
                D = feature dimension per patch
            ids_restore (torch.Tensor): Indices to restore the original order of patches, shape [B, L].

        Returns:
            torch.Tensor: Decoded tensor after transformer processing, shape [B, K, L, D].
        """        
        # embed tokens
        x = self.decoder_embed(x)

        # delete cls token
        x_ = x[:, :, 1:, :]
        B, K, L, D = x_.shape

        # detach cls token from decoder_pos_embed
        expanded_pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1).expand(B, K, -1, -1)
        expanded_cls_token = self.decoder_pos_embed[:, :1, :].unsqueeze(1).expand(B, K, -1, -1)

        # Get aligned position embeddings
        aligned_pos_embed = expanded_pos_embed.gather(dim=2, index=ids_keep_list.unsqueeze(-1).expand(-1, -1, -1, D))

        """ To make sure everything is right"""
        # # print('aligned_pos_embed', aligned_pos_embed.shape)
        # # print('expanded_pos_embed', expanded_pos_embed.shape)
        # for aligned_id, ori_id in enumerate(ids_keep_list[0, 0]):
        #     print('======================')
        #     print(aligned_id, aligned_pos_embed[0, 0, aligned_id, 0])
        #     print(ori_id, expanded_pos_embed[0, 0, ori_id, 0])
        # exit()

        aligned_pos_embed = torch.cat([expanded_cls_token, aligned_pos_embed], dim=2)

        # add aligned pos embed
        x = x + aligned_pos_embed
        # Reshape for transformer input: (N, K, L, D) -> (N * K, L, D)
        x = x.view(B * K, L + 1, D)

        # apply Transformer blocks
        i = 0
        for blk in self.decoder_blocks:
            if str(i) in self.decoder_upsample_layers:
                x = self.decoder_upsample_layers[str(i)](x)
            x = blk(x)
            i += 1

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def self_consistency_loss(self, x_p, sij):
        """
        Calculate self-consistency loss for overlapping predictions.
        
        Args:
            x_p (torch.Tensor): Predictions from different strategies, shape [N, K, L, D]
            sij (torch.Tensor): Overlap mask for each pair of strategies, shape [N, K, K, L]
        
        Returns:
            torch.Tensor: Self-consistency loss
        """
        N, K, L, D = x_p.shape

        # Stop gradients for each x_p along the K dimension for pairwise comparisons
        x_p_detach = x_p.detach()

        # Expand dimensions for broadcasting [N, K, L, D] -> [N, K, 1, L, D] and [N, 1, K, L, D]
        x_p_expanded_i = x_p.unsqueeze(2)         # [N, K, 1, L, D]
        x_p_expanded_j = x_p.unsqueeze(1)         # [N, 1, K, L, D]
        x_p_detach_i = x_p_detach.unsqueeze(2)    # [N, K, 1, L, D]
        x_p_detach_j = x_p_detach.unsqueeze(1)    # [N, 1, K, L, D]

        # Calculate pairwise consistency loss only on overlapping positions
        consistency_loss = (
            torch.abs(x_p_detach_i - x_p_expanded_j) + torch.abs(x_p_expanded_i - x_p_detach_j)
        ) * sij.unsqueeze(-1)  # Shape: [N, K, K, L, D]

        # Take the mean over the feature dimension D and length L
        consistency_loss = consistency_loss.mean(dim=(-1, -2))  # [N, K, K]

        # Only consider pairs where i < j to avoid double-counting pairs
        triu_indices = torch.triu_indices(K, K, offset=1)

        final_loss = consistency_loss[:, triu_indices[0], triu_indices[1]].mean()

        return final_loss

    def calculate_sij(self, masks):
        """
        Calculate overlapping position sij for each pair of masks.
        
        Args:
            masks (torch.Tensor): A tensor of shape [N, K, L], where
                N: batch size
                K: number of strategies
                L: length of each mask (number of patches)
        
        Returns:
            torch.Tensor: A tensor of shape [N, K, K, L] where sij[n, i, j, :]
                          represents the overlapping mask between mi and mj
                          for the nth sample.
        """
        N, K, L = masks.shape
        sij = torch.zeros((N, K, K, L), dtype=masks.dtype, device=masks.device)
        
        for i in range(K):
            for j in range(i + 1, K):
                sij[:, i, j, :] = masks[:, i, :] * masks[:, j, :]  # element-wise AND
                sij[:, j, i, :] = sij[:, i, j, :]  # Ensure symmetry
        
        return sij

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N * K, L, p*p*3]
        mask: [N, K, L], 0 is keep, 1 is remove
        """

        target = self.patchify(imgs)
        # repeat for K masking strategies
        N, K, L = mask.shape

        # [N, L, p*p*3] -> [N * K, L, p*p*3]
        target = target.unsqueeze(1).expand(N, K, L, -1).reshape(N * K, L, -1)  

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # caculate loss_whole
        loss_whole = (pred - target) ** 2
        loss_whole = loss_whole.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(N * K, L)

        loss_whole = (loss_whole * mask).sum() / mask.sum()  # mean loss on removed patches
        
        mask = mask.view(N, K, L)
        sij = self.calculate_sij(mask)
        pred = pred.view(N, K, L, -1) # Reshape pred to [N, K, L, D]

        loss_consistency = self.self_consistency_loss(pred, sij)

        total_loss = loss_whole + loss_consistency
        return total_loss



    def forward(self, imgs, mask_ratio):
        latent, mask, ids_keep_list = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_keep_list)  # [N*K, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        label = None
        return loss, pred, label, mask


@pretrained_model({})
def emae_plus_vit_base_224(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_plus_vit_large_224(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_plus_vit_base_448(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        input_size=(448, 448), patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_plus_vit_large_448(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        input_size=(448, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_plus_vit_base_512(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        input_size=(512, 512), patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_plus_vit_large_512(**kwargs):
    model = EfficientMaskedAutoencoderPlusViT(
        input_size=(512, 512), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
