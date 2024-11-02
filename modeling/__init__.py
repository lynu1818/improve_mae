# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .hiera import (
    hiera_tiny_224,
    hiera_small_224,
    hiera_base_224,
    hiera_base_plus_224,
    hiera_large_224,
    hiera_huge_224,

    hiera_tiny_512,
    hiera_base_plus_512,

    hiera_base_16x224,
    hiera_base_plus_16x224,
    hiera_large_16x224,
    hiera_huge_16x224,

    hiera_tiny_224_st_moe_50p,
    hiera_tiny_224_st_moe_0001,
    hiera_tiny_224_st_moe_0011_50p,
    

    hiera_small_224_st_moe_0011_50p,
    hiera_base_224_st_moe_0011_50p,
    hiera_base_plus_224_st_moe_0011_50p,
    hiera_large_224_st_moe_0011_50p,

    hiera_tiny_512_st_moe_0011_50p,
    hiera_base_plus_512_st_moe_0011_50p,

    Hiera,
    HieraBlock,
    HieraSTMoE,
    HieraBlockSTMoE,
    MaskUnitAttention,
    Head,
    PatchEmbed,
)


from .hiera_mae import (
    mae_hiera_tiny_224,
    mae_hiera_small_224,
    mae_hiera_base_224,
    mae_hiera_base_plus_224,
    mae_hiera_large_224,
    mae_hiera_huge_224,

    mae_hiera_tiny_512,
    mae_hiera_base_plus_512,

    mae_hiera_base_16x224,
    mae_hiera_base_plus_16x224,
    mae_hiera_large_16x224,
    mae_hiera_huge_16x224,

    mae_hiera_tiny_224_st_moe_50p,
    mae_hiera_tiny_224_st_moe_0001,
    mae_hiera_tiny_224_st_moe_0011_50p,

    mae_hiera_small_224_st_moe_0011_50p,
    mae_hiera_base_224_st_moe_0011_50p,
    mae_hiera_base_plus_224_st_moe_0011_50p,
    mae_hiera_large_224_st_moe_0011_50p,

    mae_hiera_tiny_512_st_moe_0011_50p,
    mae_hiera_base_plus_512_st_moe_0011_50p,

    MaskedAutoencoderHiera,
    MaskedAutoencoderHieraSTMoE,
)

from .hieradet import (
    hieradet_tiny_224
)

from .hieradet_mae import (
    mae_hieradet_tiny_224
)

from .hiera_abs_win import (
    HieraAbsWin,
    hiera_abs_win_tiny_224,
    hiera_abs_win_tiny_512,
    hiera_abs_win_base_plus_224,
    hiera_abs_win_base_plus_512,

    hiera_abs_win_tiny_224_st_moe_0011_50p,
    hiera_abs_win_tiny_512_st_moe_0011_50p,
    hiera_abs_win_base_plus_224_st_moe_0011_50p,
    hiera_abs_win_base_plus_512_st_moe_0011_50p,
)

from .hiera_abs_win_mae import (
    mae_hiera_abs_win_tiny_224,
    mae_hiera_abs_win_tiny_512,
    mae_hiera_abs_win_base_plus_224,
    mae_hiera_abs_win_base_plus_512,

    mae_hiera_abs_win_tiny_224_st_moe_0011_50p,
    mae_hiera_abs_win_tiny_512_st_moe_0011_50p,
    mae_hiera_abs_win_base_plus_224_st_moe_0011_50p,
    mae_hiera_abs_win_base_plus_512_st_moe_0011_50p,
)

from .vit import (
    vit_base_224,
    vit_large_224,
    vit_huge_224,
    vit_base_512,
    vit_large_512,
    VisionTransformer,
)

from .vit_mae import (
    mae_vit_base_224,
    mae_vit_large_224,
    mae_vit_huge_224,
    mae_vit_base_512,
    mae_vit_large_512,
    MaskedAutoencoderViT
)

from .vit_emae import (
    emae_vit_base_224,
    emae_vit_large_224,
    emae_vit_base_512,
    emae_vit_large_512,
    EfficientMaskedAutoencoderViT
)

from .vit_mae_plus import (
    mae_plus_vit_base_224,
    mae_plus_vit_large_224,
    mae_plus_vit_base_448,
    mae_plus_vit_large_448,
)