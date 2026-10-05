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

        # Inputs are two feature maps, the previous depth, sparse depth,
        # sparse validity, and the nine upsampled propagation logits.
        self.fusion = ConvolutionBlock(
            2 * n_channels + 12,
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
            depth_validity,
            propagation_logits,
            sparse_depth,
            sparse_validity):
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
        depth_validity = functional.interpolate(
            depth_validity,
            size=skip.shape[-2:],
            mode='nearest')
        propagation_logits = functional.interpolate(
            propagation_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        feature = self.fusion(torch.cat([
            feature,
            skip,
            depth,
            sparse_depth,
            sparse_validity,
            propagation_logits
        ], dim=1))
        propagation_logits = \
            propagation_logits + self.propagation_update(feature)

        # Valid sparse measurements replace the upsampled coarse estimate.
        source_depth = \
            sparse_validity * sparse_depth + \
            (1.0 - sparse_validity) * depth
        source_validity = torch.clamp(
            sparse_validity + depth_validity,
            min=0.0,
            max=1.0)
        depth, depth_validity = propagate_depth(
            depth=source_depth,
            validity_map=source_validity,
            propagation_logits=propagation_logits)

        # Propagation can only mix existing depth values. This learned
        # residual lets the RGB-sparse feature restore spatial detail that
        # was not represented by the coarse sparse-depth grid.
        depth = depth + self.depth_residual(feature)
        depth_validity = torch.ones_like(depth_validity)
        return \
            feature, depth, depth_validity, propagation_logits


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

        # Predict the reliability of every input sparse-depth measurement.
        # The confidence is only used where sparse depth is valid.
        self.sparse_confidence = Conv2d(
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

    def downsample_sparse_depth(self, sparse_depth, confidence, size):
        '''Confidence-weight valid sparse points inside every output cell.'''
        validity_map = (sparse_depth > 0.0).to(sparse_depth.dtype)
        pooled_depth = functional.adaptive_avg_pool2d(
            sparse_depth * validity_map * confidence,
            output_size=size)
        pooled_confidence = functional.adaptive_avg_pool2d(
            validity_map * confidence,
            output_size=size)
        pooled_validity = functional.adaptive_avg_pool2d(
            validity_map,
            output_size=size)
        downsampled_depth = pooled_depth / (pooled_confidence + 1e-7)
        downsampled_confidence = \
            pooled_confidence / (pooled_validity + 1e-7)
        return \
            downsampled_depth * (pooled_validity > 0.0), \
            downsampled_confidence

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)
        sparse_confidence = torch.sigmoid(
            self.sparse_confidence(features[0]))

        coarse_feature = features[-1]
        coarse_sparse_depth, coarse_validity = self.downsample_sparse_depth(
            sparse_depth,
            sparse_confidence,
            coarse_feature.shape[-2:])
        propagation_logits = self.propagation_map(coarse_feature)
        depth, depth_validity = propagate_depth(
            depth=coarse_sparse_depth,
            validity_map=coarse_validity,
            propagation_logits=propagation_logits)

        feature = coarse_feature
        for decoder_block, level in zip(
                self.decoder,
                range(3, -1, -1)):
            level_sparse_depth, level_sparse_validity = \
                self.downsample_sparse_depth(
                    sparse_depth,
                    sparse_confidence,
                    features[level].shape[-2:])
            feature, depth, depth_validity, propagation_logits = \
                decoder_block(
                feature=feature,
                skip=features[level],
                depth=depth,
                depth_validity=depth_validity,
                propagation_logits=propagation_logits,
                sparse_depth=level_sparse_depth,
                sparse_validity=level_sparse_validity)

        depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return depth
