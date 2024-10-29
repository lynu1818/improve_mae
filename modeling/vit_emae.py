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

from torch.profiler import profile, ProfilerActivity, record_function

class EfficientMaskedAutoencoderViT(nn.Module):
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
        for i in range(num_mask_strategy):
            # Calculate start and end indices for the i-th subset
            start_idx = i * (L // num_mask_strategy)
            end_idx = (i + 1) * (L // num_mask_strategy)
            
            # Obtain the i-th visible patches
            ids_keep_i = ids_shuffle[:, start_idx:end_idx]
            x_visible_patches[:, i] = torch.gather(x, dim=1, index=ids_keep_i.unsqueeze(-1).repeat(1, 1, D))

            # Obtain the i-th mask
            m_i = torch.ones([N, L], device=x.device)
            m_i[:, start_idx:end_idx] = 0
            # Unshuffle to get the binary mask
            mask_strategies[:, i] = torch.gather(m_i, dim=1, index=ids_restore)
        
        
        # x_visible_patches = torch.stack(x_visible_patches, dim=1)
        # mask_strategies = torch.stack(mask_strategies, dim=1)

        # x_visible_patches: (N, K ,L//K, D)
        # mask_strategies: (N, K, L)
        # ids_restore: (N, L)
        # print(f'random masking:')
        # print(f'N: {N}, K: {num_mask_strategy}, L: {L}, L//K: {L//num_mask_strategy}, D: {D}')
        # print(f' x visible patches shape: {x_visible_patches.shape}')
        # print(f' mask_strategies shape: {mask_strategies.shape}')
        # print(f' ids_restore shape: {ids_restore.shape}')
        return x_visible_patches, mask_strategies, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # print(f'forward encdoer')
        # print(f'x shape: {x.shape}, mask ratio: {mask_ratio}')
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

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

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Args:
            x (torch.Tensor): Input tensor after encoding, shape [N, K, L+1, D], where:
                N = batch size
                K = number of masking strategies
                L = number of patches per strategy
                D = feature dimension per patch
            ids_restore (torch.Tensor): Indices to restore the original order of patches, shape [N, L].

        Returns:
            torch.Tensor: Decoded tensor after transformer processing, shape [N, K, L, D].
        """
        # print(f'forward decoder')
        # print(f'x shape: {x.shape}, ids_restore shape: {ids_restore.shape}')
        
        # embed tokens
        x = self.decoder_embed(x)


        # delete cls token
        x_ = x[:, :, 1:, :]

        N, K, L, D = x_.shape
        # print(f'x_ shape: {x_.shape}')
        ids_restore = ids_restore.repeat_interleave(K, dim=0)
        # print(f'new ids restore shape: {ids_restore.shape}')
        # print(f'self num patches: {self.num_patches}')
        

        x_with_masks = torch.zeros((N, K, self.num_patches, D), device=x_.device)
        # print(f'x_with_mask shape: {x_with_masks.shape}')
        
        for i in range(K):
            start_idx = i * L
            end_idx = (i + 1) * L

            x_with_masks[:, i, start_idx:end_idx, :] = x_[:, i, :, :]


        # Reshape for transformer input: (N, K, L, D) -> (N * K, L, D)
        x_ = x_with_masks.view(N * K, self.num_patches, D)
        # print(f'x_ shape2: {x_.shape}')

        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D).to(x_.device))
        # print(f'x_ shape3: {x_.shape}')
        # print(f'decoder pos embed shape: {self.decoder_pos_embed.shape}')

        x_ = x_.to(x.device)
        x = x.view(N * K, L + 1, D)

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
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

        # print(f'forward loss')
        # print(f'target shape: {target.shape}, pred shape: {pred.shape}, mask shape: {mask.shape}')


        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # caculate loss_whole
        loss_whole = (pred - target) ** 2
        loss_whole = loss_whole.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(N * K, L)
        # print(f'new mask shape: {mask.shape}')
        loss_whole = (loss_whole * mask).sum() / mask.sum()  # mean loss on removed patches
        
        mask = mask.view(N, K, L)
        sij = self.calculate_sij(mask)
        pred = pred.view(N, K, L, -1) # Reshape pred to [N, K, L, D]
        # print(f'new pred shape: {pred.shape}')
        loss_consistency = self.self_consistency_loss(pred, sij)

        total_loss = loss_whole + loss_consistency
        return total_loss



    def forward(self, imgs, mask_ratio):
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        #     on_trace_ready=trace_handler,
        #     record_shapes=True
        # ) as prof:
        #     with record_function("encoder"):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            # with record_function("decoder"):
        pred = self.forward_decoder(latent, ids_restore)  # [N*K, L, p*p*3]
            # with record_function("calculate loss"):
        loss = self.forward_loss(imgs, pred, mask)
            # prof.step()

        label = None
        return loss, pred, label, mask

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sort_by_keyword = "self_cuda_time_total"

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    
    print("Profiler output:", output)
    
    with open("/mnt/home/andyqmongo/lynu369/hiera_moe/profiler/profiler_output3.txt", "a") as f:
        f.write("Step: " + str(p.step_num) + "\n")
        f.write(output)
        f.write("\n\n")

    p.export_chrome_trace(f"/mnt/home/andyqmongo/lynu369/hiera_moe/profiler/trace_step_{p.step_num}.json")


@pretrained_model({})
def emae_vit_base_224(**kwargs):
    model = EfficientMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_vit_large_224(**kwargs):
    model = EfficientMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_vit_base_512(**kwargs):
    model = EfficientMaskedAutoencoderViT(
        input_size=(512, 512), patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def emae_vit_large_512(**kwargs):
    model = EfficientMaskedAutoencoderViT(
        input_size=(512, 512), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
