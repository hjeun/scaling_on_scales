#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard, get_2d_sincos_pos_embed

### training for scaling on scales

# img_sizes = [448, 1344]
# max_split_size = 448
# bs = images.shape[0]
# multiscale_images, num_splits = create_multiscale_images(images,
#                                                          img_sizes=image_sizes,
#                                                          max_split_size=max_split_size)
# multiscale_feats = model(multiscale_images)
# image_feats = merge_mutliscale_features(multiscale_feats, bs, num_splits)

def create_multiscale_images(input, img_sizes=[448, 1344], max_split_size=448):
  
    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    
    bs, c, input_size, _ = input.shape

    # image size for each scale
    assert img_sizes is not None, "Please assign either scales or img_sizes."
    assert len(img_sizes) == 2, "Currently the supported number of image sizes is only 2."

    # prepare multiscale inputs
    max_split_size = max_split_size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        if size != input_size:
            x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        else:
            x = input
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)
    multiscale_images = torch.cat(input_multiscale)
    return multiscale_images, num_splits


def merge_mutliscale_features(multiscale_features, bs, num_splits, resize_output_to_idx=0, num_prefix_token=0, output_shape='bnc', use_pos_embed=False):
    outs_multiscale = [multiscale_features[:bs], multiscale_features[bs:]]
    
    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                           for out in outs_multiscale]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in zip(num_splits, outs_multiscale)]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    
    outs = []
    for i in range(len(outs_multiscale)):
        if outs_multiscale[i].shape[2] != output_size:
            out = F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
        else:        
            out = outs_multiscale[i]
        outs.append(out)
    outs = torch.cat(outs, dim=1)

    if output_shape == 'bnc':
        outs = rearrange(outs, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [torch.stack(outs.split(bs, dim=0), dim=0).mean(dim=0) for outs in outs_prefix_multiscale]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        outs = torch.cat([out_prefix_multiscale, outs], dim=1)

    if use_pos_embed:
        num_patches = outs.shape[1]
        embed_dim = outs.shape[2]
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(pos_embed.shape[-1],
                                            int(num_patches**.5),
                                            cls_token=False)
        outs = outs + pos_embed
    return outs


def forward(model, input, scales=None, img_sizes=None, max_split_size=None, resize_output_to_idx=0, num_prefix_token=0,
            output_shape='bnc'):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    assert output_shape in ['bnc', 'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # image size for each scale
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size   # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [model(x) for x in input_multiscale]
    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                           for out in outs_multiscale]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in zip(num_splits, outs_multiscale)]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in outs_prefix_multiscale]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out
