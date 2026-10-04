import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as functional

sys.path.insert(0, os.path.join(
    'external_src', 'depth_completion', 'kbnet', 'src'))
from net_utils import Conv2d


def add_position_encoding(feature):
    '''Add normalized two-dimensional Fourier position information.'''
    _, n_channel, n_height, n_width = feature.shape
    n_frequency = n_channel // 4

    y = (torch.arange(
        n_height,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_height
    x = (torch.arange(
        n_width,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_width
    frequencies = 2.0 ** torch.arange(
        n_frequency,
        dtype=feature.dtype,
        device=feature.device)

    y_angle = 2.0 * math.pi * y[:, None] * frequencies[None, :]
    x_angle = 2.0 * math.pi * x[:, None] * frequencies[None, :]
    position = torch.cat([
        torch.sin(y_angle)[:, None, :].expand(
            n_height, n_width, n_frequency),
        torch.cos(y_angle)[:, None, :].expand(
            n_height, n_width, n_frequency),
        torch.sin(x_angle)[None, :, :].expand(
            n_height, n_width, n_frequency),
        torch.cos(x_angle)[None, :, :].expand(
            n_height, n_width, n_frequency)
    ], dim=2)
    position = position.permute(2, 0, 1)[None, :, :, :]
    return feature + position


class SelfAttentionBlock(nn.Module):
    '''One residual self-attention and feedforward block.'''

    def __init__(self, n_channels, n_head):
        super(SelfAttentionBlock, self).__init__()

        self.attention_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            batch_first=True)
        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Linear(4 * n_channels, n_channels))

    def forward(self, tokens):
        normalized_tokens = self.attention_norm(tokens)
        attended_tokens, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False)
        tokens = tokens + attended_tokens
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class ConvolutionBlock(nn.Module):
    '''Three spatial convolutions; the first optionally downsamples.'''

    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionBlock, self).__init__()

        activation = torch.nn.LeakyReLU(
            negative_slope=0.10,
            inplace=True)
        self.convolutions = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                activation_func=activation),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(
                    negative_slope=0.10,
                    inplace=True)),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(
                    negative_slope=0.10,
                    inplace=True)))

    def forward(self, feature):
        return self.convolutions(feature)


class SharedEncoder(nn.Module):
    '''Five-resolution encoder shared by concatenated RGB and sparse depth.'''

    def __init__(self, n_channels, n_head):
        super(SharedEncoder, self).__init__()

        self.level0 = ConvolutionBlock(4, n_channels, stride=1)
        self.level1 = ConvolutionBlock(n_channels, n_channels, stride=2)
        self.level2 = ConvolutionBlock(n_channels, n_channels, stride=2)

        # Only the resolution-changing convolution remains at the last two
        # levels; spatial processing is performed by two attention blocks.
        self.level3_downsample = Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            activation_func=torch.nn.LeakyReLU(
                negative_slope=0.10,
                inplace=True))
        self.level4_downsample = Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            activation_func=torch.nn.LeakyReLU(
                negative_slope=0.10,
                inplace=True))
        self.level3_attention = nn.ModuleList([
            SelfAttentionBlock(n_channels, n_head)
            for _ in range(2)
        ])
        self.level4_attention = nn.ModuleList([
            SelfAttentionBlock(n_channels, n_head)
            for _ in range(2)
        ])

    def attention_level(self, feature, attention_blocks):
        n_batch, n_channel, n_height, n_width = feature.shape
        feature = add_position_encoding(feature)
        tokens = feature.permute(0, 2, 3, 1).reshape(
            n_batch,
            n_height * n_width,
            n_channel)
        for attention_block in attention_blocks:
            tokens = attention_block(tokens)
        return tokens.reshape(
            n_batch,
            n_height,
            n_width,
            n_channel).permute(0, 3, 1, 2)

    def forward(self, image, sparse_depth):
        feature = torch.cat([image, sparse_depth], dim=1)

        level0 = self.level0(feature)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        level3 = self.attention_level(
            self.level3_downsample(level2),
            self.level3_attention)
        level4 = self.attention_level(
            self.level4_downsample(level3),
            self.level4_attention)

        return [level0, level1, level2, level3, level4]


class MultiscaleDecoderBlock(nn.Module):
    '''Upsample and fuse one encoder skip and the preceding depth estimate.'''

    def __init__(self, n_channels, min_predict_depth):
        super(MultiscaleDecoderBlock, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.fusion = ConvolutionBlock(
            2 * n_channels + 1,
            n_channels,
            stride=1)
        self.depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(self, feature, skip, depth):
        feature = functional.interpolate(
            feature,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        depth = functional.interpolate(
            depth,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        feature = self.fusion(torch.cat([
            feature,
            skip,
            depth
        ], dim=1))
        depth = self.min_predict_depth + functional.softplus(
            self.depth(feature))
        return feature, depth


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder, bottom propagation map, and multiscale decoder.'''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # The positive map directly scales valid sparse depth at the bottom.
        self.propagation_map = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

        self.decoder = nn.ModuleList([
            MultiscaleDecoderBlock(
                n_channels=n_channels,
                min_predict_depth=min_predict_depth)
            for _ in range(4)
        ])

    def downsample_sparse_depth(self, sparse_depth, size):
        '''Average only valid sparse points inside every output cell.'''
        validity_map = (sparse_depth > 0.0).to(sparse_depth.dtype)
        pooled_depth = functional.adaptive_avg_pool2d(
            sparse_depth * validity_map,
            output_size=size)
        pooled_validity = functional.adaptive_avg_pool2d(
            validity_map,
            output_size=size)
        downsampled_depth = pooled_depth / (pooled_validity + 1e-7)
        downsampled_validity = \
            (pooled_validity > 0.0).to(sparse_depth.dtype)
        return \
            downsampled_depth * downsampled_validity, \
            downsampled_validity

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)

        coarse_feature = features[-1]
        coarse_sparse_depth, _ = self.downsample_sparse_depth(
            sparse_depth,
            coarse_feature.shape[-2:])
        propagation_map = functional.softplus(
            self.propagation_map(coarse_feature))
        coarse_depth = propagation_map * coarse_sparse_depth

        feature = coarse_feature
        depth = coarse_depth
        for decoder_block, level in zip(
                self.decoder,
                range(3, -1, -1)):
            feature, depth = decoder_block(
                feature=feature,
                skip=features[level],
                depth=depth)

        depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return [depth, coarse_depth]
