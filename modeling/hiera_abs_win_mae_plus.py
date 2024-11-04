# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae
# slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------


from functools import partial
from typing import Tuple, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hiera import HieraBlock
from .hiera_abs_win import HieraAbsWin
from .hiera_utils import pretrained_model, undo_windowing, conv_nd
from .hfhub import has_config

def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


# class DecoderUpsampleBlock(nn.Module):
#     def __init__(self, D, upscale_factor):
#         super().__init__()
#         self.upscale_factor = upscale_factor
#         self.conv = nn.Conv2d(D, D * (upscale_factor ** 2), kernel_size=3, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
#     def forward(self, x):
#         # No cls token
#         B, N, D = x.shape
#         H = W = int(N ** .5)
#         x = x.view(B, H, W, D).permute(0, 3, 1, 2) # B, D, H, W
#         # Upsample
#         x = self.conv(x) # B, D * (upscale_factor ** 2), H, W
#         x = self.pixel_shuffle(x) # B, D, H * upscale_factor, W * upscale_factor

#         # Flatten
#         x = x.permute(0, 2, 3, 1).reshape(B, -1, D)
#         return x


class TokenUpsampleBlock(nn.Module):
    def __init__(self, dim, input_length, target_length):
        super().__init__()
        self.proj = nn.Linear(dim, (target_length // input_length) * dim)
        self.target_length = target_length
    
    def forward(self, x):
        # x shape: [B, L, D]
        B, L, D = x.shape
        x = self.proj(x)  # Shape: [B, L, r*D]
        x = x.view(B, self.target_length, D)  # Reshape to [B, rL, D]
        return x



class MaskedAutoencoderPlusHieraAbsWin(HieraAbsWin):
    """Masked Autoencoder with Hiera backbone with window pos embed"""

    @has_config
    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        **kwdargs,
    ):
        super().__init__(
            #in_chans=in_chans,
            patch_stride=patch_stride,
            #mlp_ratio=mlp_ratio,
            #norm_layer=norm_layer,
            **kwdargs,
        )
        
        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        self.num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        # to be append if needed. e.g. res=224, ratio=0.75, #tokens=48 -> need to extend to 49
        #TODO: Add support for other resolutions
        if self.input_size[0] == 224:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) 

        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_upsample_layers = None
        if decoder_depth == 8:
            if self.mask_ratio == 0.75:
                self.decoder_upsample_layers = nn.ModuleDict({
                    str(decoder_depth - 2): TokenUpsampleBlock(decoder_embed_dim, self.num_windows, self.num_windows*4),
                })
        if self.decoder_upsample_layers is None:
            raise ValueError(f"Unsupported mask ratio {self.mask_ratio} for decoder depth {decoder_depth}")

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, ::self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(0, 2, 3, 4, 5, 6, 1)  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        len_keep = int(self.num_windows * (1 - mask_ratio))
        noise = torch.rand(B, self.num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, self.num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        ids_keep = ids_shuffle[:, :len_keep]
        return mask.bool(), ids_keep

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mask, ids_keep = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)
        x = self.encoder_norm(x)
        return x, mask, ids_keep

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor, ids_keep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)
        B, D = x.shape[0], x.shape[-1]

        x = x.view(B, -1, D)
        mask_spatial_size =self.mask_spatial_shape[0]
        mask = mask.view(B, mask_spatial_size, mask_spatial_size)
        mask = mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        mask_flat = mask.view(B, -1)
        indicies = mask_flat.nonzero(as_tuple=True)
        selected_pos_embeddings = self.decoder_pos_embed[:, indicies[1], :]
        selected_pos_embeddings = selected_pos_embeddings.view(B, -1, D)

        # Add pos embed
        x = x + selected_pos_embeddings

        if x.shape[1] != self.num_windows:
            diff = self.num_windows - x.shape[1]
            x = torch.cat([x, self.mask_token.expand(B, diff, D)], dim=1)
        # Apply decoder blocks
        i = 0
        for blk in self.decoder_blocks:
            if str(i) in self.decoder_upsample_layers:
                x = self.decoder_upsample_layers[str(i)](x)
            x = blk(x)
            i += 1
        
        x = self.decoder_norm(x)
        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask.view(B, -1)

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latent, mask, ids_shuffle = self.forward_encoder(x, mask_ratio)
        pred, pred_mask = self.forward_decoder(
            latent, mask, ids_shuffle
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask


@pretrained_model({})
def mae_plus_hiera_abs_win_tiny_224(**kwargs):
    return MaskedAutoencoderPlusHieraAbsWin(
        embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), q_pool=2, **kwargs,
    )

@pretrained_model({})
def mae_plus_hiera_abs_win_tiny_448(**kwargs):
    return MaskedAutoencoderPlusHieraAbsWin(
        embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), q_pool=2, input_size=(448, 448), **kwargs,
    )

@pretrained_model({})
def mae_plus_hiera_abs_win_base_plus_224(**kwargs):
    return MaskedAutoencoderPlusHieraAbsWin(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), q_pool=2, **kwargs,
    )

@pretrained_model({})
def mae_plus_hiera_abs_win_base_plus_448(**kwargs):
    return MaskedAutoencoderPlusHieraAbsWin(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), q_pool=2, input_size=(448, 448), **kwargs,
    )
