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

from .vit_mae_util.pos_embed import get_2d_sincos_pos_embed, get_spatial_2d_sincos_pos_embed
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

class TokenUpsampleBlock(nn.Module):
    def __init__(self, dim, input_length, target_length):
        super().__init__()
        self.proj = nn.Linear(dim, (target_length // input_length) * dim)
        self.target_length = target_length
    
    def forward(self, x):
        # remove cls token
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        # x shape: [B, L, D]
        B, L, D = x.shape
        x = self.proj(x)  # Shape: [B, L, r*D]
        x = x.view(B, self.target_length, D)  # Reshape to [B, rL, D]
        x = torch.cat([cls_token, x], dim=1)
        return x

class MaskedAutoencoderPlusViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    @has_config
    def __init__(self, mask_ratio, model_name = '', input_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 drop_path_rate: float = 0.0, upsample_method='pixel_shuffle', dec_pos_embed_type='standard_sincos'):
        super().__init__()

        self.model_name = model_name
        img_size = input_size[0]
        self.input_size = input_size
        self.dec_pos_embed_type = dec_pos_embed_type
        print(f"======== configs ==========")
        print(f"model_name: {model_name}")
        print(f"mask_ratio: {mask_ratio}")
        print(f"input_size: {input_size}")
        print('upsample_method:', upsample_method)
        print('dec_pos_embed_type:', dec_pos_embed_type)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

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

        if self.dec_pos_embed_type.find('learnable') >= 0:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)  # Learnable pos embedding
        else:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # Non-Learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_upsample_layers = None
        
        if upsample_method == 'token':
            if mask_ratio == 0.75:
                if img_size == 224:
                    self.decoder_upsample_layers = nn.ModuleDict({
                        str(decoder_depth - 2): TokenUpsampleBlock(decoder_embed_dim, 49, 196),
                    })
                elif img_size == 448:
                    self.decoder_upsample_layers = nn.ModuleDict({
                        str(decoder_depth - 2): TokenUpsampleBlock(decoder_embed_dim, 196, 784),
                    })
        elif upsample_method == 'pixel_shuffle':
            if decoder_depth == 8:
                if mask_ratio == 0.75:
                    self.decoder_upsample_layers = nn.ModuleDict({
                        str(decoder_depth - 2): DecoderUpsampleBlock(decoder_embed_dim, 2),
                    })
                elif mask_ratio == 0.9375:
                    self.decoder_upsample_layers = nn.ModuleDict({
                        str(decoder_depth - 2): DecoderUpsampleBlock(decoder_embed_dim, 2),
                        str(decoder_depth - 1): DecoderUpsampleBlock(decoder_embed_dim, 2)
                    })
            elif decoder_depth == 2:
                if mask_ratio == 0.75:
                    self.decoder_upsample_layers = nn.ModuleDict({
                        str(decoder_depth - 1): DecoderUpsampleBlock(decoder_embed_dim, 2),
                    })
        if self.decoder_upsample_layers is None:
            raise ValueError(f"Unsupported mask ratio {mask_ratio} for decoder depth {decoder_depth}")

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

        if self.dec_pos_embed_type.find('random') >= 0:
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        elif self.dec_pos_embed_type.find('spatial_sincos') >= 0:
            decoder_pos_embed = get_spatial_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        elif self.dec_pos_embed_type.find('standard_sincos') >= 0:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            raise ValueError(f"Unsupported decoder pos embed type {self.dec_pos_embed_type}")

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_shuffle

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_shuffle = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_shuffle

    def forward_decoder(self, x, ids_shuffle):
        # embed tokens
        x = self.decoder_embed(x)

        B, L, D = x.shape
        L -= 1  # remove cls token
        ids_keep = ids_shuffle[:, :L]

        # detach cls token from decoder_pos_embed
        expanded_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(B, -1, -1)
        expanded_cls_token = self.decoder_pos_embed[:, :1, :].expand(B, -1, -1)

        # Get aligned position embeddings
        aligned_pos_embed = expanded_pos_embed.gather(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        aligned_pos_embed = torch.cat([expanded_cls_token, aligned_pos_embed], dim=1)

        # add aligned position embeddings to input
        x = x + aligned_pos_embed

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

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio):
        latent, mask, ids_shuffle = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_shuffle)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        label = None
        return loss, pred, label, mask


@pretrained_model({})
def mae_plus_vit_base_224(**kwargs):
    model = MaskedAutoencoderPlusViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def mae_plus_vit_large_224(**kwargs):
    model = MaskedAutoencoderPlusViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def mae_plus_vit_huge_224(**kwargs):
    model = MaskedAutoencoderPlusViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def mae_plus_vit_base_448(**kwargs):
    model = MaskedAutoencoderPlusViT(
        input_size=(448, 448), patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@pretrained_model({})
def mae_plus_vit_large_448(**kwargs):
    model = MaskedAutoencoderPlusViT(
        input_size=(448, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
