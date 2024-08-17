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

    MaskedAutoencoderHiera,
    MaskedAutoencoderHieraSTMoE,
)