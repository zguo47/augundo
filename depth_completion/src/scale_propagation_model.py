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

        self.convolutions = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)))

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
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True))
        self.level4_downsample = Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True))
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


def propagate_depth(depth, validity_map, propagation_logits):
    '''Multiply each local depth neighborhood by learned propagation weights.'''
    n_batch, _, n_height, n_width = depth.shape
    propagation_weights = torch.softmax(propagation_logits, dim=1)

    depth_patches = functional.unfold(
        functional.pad(depth, (1, 1, 1, 1), mode='replicate'),
        kernel_size=3).reshape(
            n_batch, 9, n_height, n_width)
    validity_patches = functional.unfold(
        functional.pad(validity_map, (1, 1, 1, 1), mode='replicate'),
        kernel_size=3).reshape(
            n_batch, 9, n_height, n_width)

    valid_weights = propagation_weights * validity_patches
    weight_sum = torch.sum(valid_weights, dim=1, keepdim=True)
    propagated_depth = torch.sum(
        valid_weights * depth_patches,
        dim=1,
        keepdim=True) / (weight_sum + 1e-7)
    propagated_validity = (weight_sum > 0.0).to(depth.dtype)
    return \
        propagated_depth * propagated_validity, \
        propagated_validity


class PropagationDecoderBlock(nn.Module):
    '''Upsample and refine the preceding propagation map at one scale.'''

    def __init__(self, n_channels):
        super(PropagationDecoderBlock, self).__init__()

        # Inputs are two feature maps, the previous dense depth, and the nine
        # upsampled propagation logits.
        self.fusion = ConvolutionBlock(
            2 * n_channels + 10,
            n_channels,
            stride=1)
        self.propagation_update = Conv2d(
            n_channels,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.depth_residual = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(
            self,
            feature,
            skip,
            depth,
            propagation_logits):
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
        propagation_logits = functional.interpolate(
            propagation_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        feature = self.fusion(torch.cat([
            feature,
            skip,
            depth,
            propagation_logits
        ], dim=1))
        propagation_logits = \
            propagation_logits + self.propagation_update(feature)

        # The residual creates new spatial detail instead of restricting the
        # finer depth to mixtures of the preceding coarse values.
        depth = depth + self.depth_residual(feature)
        depth, _ = propagate_depth(
            depth=depth,
            validity_map=torch.ones_like(depth),
            propagation_logits=propagation_logits)
        return \
            feature, depth, propagation_logits


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

        # The bottom feature predicts a dense coarse depth. Sparse depth is
        # not pooled into the depth state.
        self.coarse_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

        # Nine logits define one learned 3 x 3 propagation map per pixel.
        self.propagation_map = Conv2d(
            n_channels,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)

        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(n_channels=n_channels)
            for _ in range(4)
        ])

        # Confidence determines how strongly each original-resolution sparse
        # measurement constrains the learned dense prediction.
        self.sparse_confidence = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)

        coarse_feature = features[-1]
        depth = functional.softplus(
            self.coarse_depth(coarse_feature)) + self.min_predict_depth
        propagation_logits = self.propagation_map(coarse_feature)
        depth, _ = propagate_depth(
            depth=depth,
            validity_map=torch.ones_like(depth),
            propagation_logits=propagation_logits)

        feature = coarse_feature
        for decoder_block, level in zip(
                self.decoder,
                range(3, -1, -1)):
            feature, depth, propagation_logits = \
                decoder_block(
                feature=feature,
                skip=features[level],
                depth=depth,
                propagation_logits=propagation_logits)

        # Apply metric constraints only at their original pixel locations.
        sparse_validity = (sparse_depth > 0.0).to(sparse_depth.dtype)
        sparse_confidence = \
            torch.sigmoid(self.sparse_confidence(feature)) * sparse_validity
        depth = \
            sparse_confidence * sparse_depth + \
            (1.0 - sparse_confidence) * depth
        depth, _ = propagate_depth(
            depth=depth,
            validity_map=torch.ones_like(depth),
            propagation_logits=propagation_logits)

        depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return depth
