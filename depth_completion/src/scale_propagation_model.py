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
    '''Self-attention and feedforward block.'''

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

        # For the last two levels, there is one downsampling conv + 2 attention blocks.
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


def normalize_scaleless_depth(scaleless_depth):
    '''Remove the global scale while preserving the predicted spatial shape.'''

    mean_depth = torch.mean(
        scaleless_depth,
        dim=[2, 3],
        keepdim=True)
    return scaleless_depth / (mean_depth + 1e-7)


def sparse_depth_at_resolution(sparse_depth, output_size):
    '''Average only valid sparse measurements inside each output cell.'''

    sparse_validity = (sparse_depth > 0.0).to(sparse_depth.dtype)
    pooled_depth = functional.adaptive_avg_pool2d(
        sparse_depth,
        output_size=output_size)
    pooled_validity = functional.adaptive_avg_pool2d(
        sparse_validity,
        output_size=output_size)
    sparse_depth = pooled_depth / (pooled_validity + 1e-7)
    sparse_validity = (pooled_validity > 0.0).to(sparse_depth.dtype)
    return sparse_depth * sparse_validity, sparse_validity


def sparse_scale_at_resolution(sparse_depth, scaleless_depth):
    '''Convert sparse metric depth into sparse scale.'''

    sparse_depth, sparse_validity = sparse_depth_at_resolution(
        sparse_depth=sparse_depth,
        output_size=scaleless_depth.shape[-2:])
    sparse_scale = sparse_depth / torch.clamp(
        scaleless_depth,
        min=1e-3)
    return sparse_scale * sparse_validity, sparse_validity


def propagate_scale(scale, propagation_logits):
    '''Propagate a scale field without averaging the scaleless depth map.'''

    n_batch, _, n_height, n_width = scale.shape
    # Convert to weights
    propagation_weights = torch.softmax(propagation_logits, dim=1)

    scale_patches = functional.unfold(
        functional.pad(scale, (1, 1, 1, 1), mode='replicate'),
        kernel_size=3).reshape(n_batch, 9, n_height, n_width)

    return torch.sum(
        propagation_weights * scale_patches,
        dim=1,
        keepdim=True)


def rejection_keep_mask(rejection_logits, temperature, training):
    '''Convert two-channel keep/reject logits into a hard keep mask.'''

    if training:
        rejection_decision = functional.gumbel_softmax(
            rejection_logits,
            tau=temperature,
            hard=True,
            dim=1)
        return rejection_decision[:, 0:1, :, :]

    rejection_decision = torch.argmax(
        rejection_logits,
        dim=1,
        keepdim=True)
    return (rejection_decision == 0).to(rejection_logits.dtype)


class PropagationDecoderBlock(nn.Module):
    '''Refine scaleless depth and propagate metric scale at one resolution.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection):
        super(PropagationDecoderBlock, self).__init__()

        # Inputs are decoder features, encoder skip features, scaleless depth,
        # metric scale, nine propagation logits, and two rejection logits.
        # C + C + 1 + 1 + 9 + 2 = 2C + 13
        self.fusion = ConvolutionBlock(
            2 * n_channels + 13,
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.propagation_update = Conv2d(
            n_channels,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_update = Conv2d(
            n_channels,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scaleless_depth_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(
            self,
            feature,
            skip,
            scaleless_depth,
            scale,
            sparse_depth,
            propagation_logits,
            rejection_logits):
        # Upsample all coarse quantities.
        feature = functional.interpolate(
            feature,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        scaleless_depth = functional.interpolate(
            scaleless_depth,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        scale = functional.interpolate(
            scale,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        propagation_logits = functional.interpolate(
            propagation_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        rejection_logits = functional.interpolate(
            rejection_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        # Fuse with fine encoder features.
        feature = self.fusion(torch.cat([
            feature,
            skip,
            scaleless_depth,
            scale,
            propagation_logits,
            rejection_logits
        ], dim=1))
        propagation_logits = propagation_logits + self.propagation_update(feature)
        if self.use_outlier_rejection:
            rejection_logits = rejection_logits + self.rejection_update(feature)
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            rejection_logits = torch.zeros_like(rejection_logits)
            keep_mask = torch.ones_like(scaleless_depth)

        # Preserves positivity and prevent huge negative residual.
        scaleless_depth = scaleless_depth * torch.exp(torch.tanh(
            self.scaleless_depth_update(feature)))
        # Normalize to avoid introducing unwanted scale.
        scaleless_depth = normalize_scaleless_depth(scaleless_depth)

        # Compute sparse scale and validity at the current resolution.
        sparse_scale, sparse_validity = sparse_scale_at_resolution(
            sparse_depth=sparse_depth,
            scaleless_depth=scaleless_depth)
        # Select accepted sparse depth locations
        accepted_sparse_validity = sparse_validity * keep_mask
        # Apply scale at accepted locations
        scale = accepted_sparse_validity * sparse_scale + \
            (1.0 - accepted_sparse_validity) * scale
        # Propagate scale.
        scale = propagate_scale(
            scale=scale,
            propagation_logits=propagation_logits)

        # Keep accepted sparse depth anchors exact after the local scale propagation.
        scale = accepted_sparse_validity * sparse_scale + \
            (1.0 - accepted_sparse_validity) * scale
        return feature, \
            scaleless_depth, \
            scale, \
            propagation_logits, \
            rejection_logits, \
            keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder and coarse-to-fine scale propagation decoder.'''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0,
                 use_outlier_rejection=True):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection

        # Takes in RGB and sparse depth together and produce 5 feature maps.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # The bottom feature predicts depth shape without metric scale.
        self.coarse_scaleless_depth = Conv2d(
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

        # Two logits represent keep and reject for every coarse depth.
        self.rejection_map = Conv2d(
            n_channels,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        # Initialize with a preference for keeping depth values.
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=gumbel_temperature,
                use_outlier_rejection=use_outlier_rejection)
            for _ in range(4)
        ])

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)

        coarse_feature = features[-1]
        scaleless_depth = functional.softplus(
            self.coarse_scaleless_depth(coarse_feature)) + 1e-3
        scaleless_depth = normalize_scaleless_depth(scaleless_depth)
        # B x 9 x H_4 x W_4 scale propagation logits.
        propagation_logits = self.propagation_map(coarse_feature)
        if self.use_outlier_rejection:
            # Predict which sparse depth points to keep.
            rejection_logits = \
                self.rejection_map(coarse_feature) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            # Keep the decoder input shape unchanged while accepting all
            # sparse depth points.
            rejection_logits = torch.zeros_like(
                propagation_logits[:, 0:2, :, :])
            keep_mask = torch.ones_like(scaleless_depth)

        # Convert sparse depth to sparse scale.
        sparse_scale, sparse_validity = sparse_scale_at_resolution(
            sparse_depth=sparse_depth,
            scaleless_depth=scaleless_depth)
        # Compute initial global scale B x 1 x 1 x 1.
        scale = torch.sum(
            sparse_scale * sparse_validity,
            dim=[2, 3],
            keepdim=True) / (
                torch.sum(
                    sparse_validity,
                    dim=[2, 3],
                    keepdim=True) + 1e-7)
        # Expand the initial global scale to match the spatial dimensions of the scaleless depth.
        scale = scale.expand_as(scaleless_depth)

        # Apply keep_mask to locations of accepted sparse depth.
        accepted_sparse_validity = sparse_validity * keep_mask
        # Accepted locations use measured scale. Others use global scale.
        scale = accepted_sparse_validity * sparse_scale + \
            (1.0 - accepted_sparse_validity) * scale
        # Propagate scale locally.
        scale = propagate_scale(
            scale=scale,
            propagation_logits=propagation_logits)
        # Restore accepted location to be observed scales.
        scale = accepted_sparse_validity * sparse_scale + \
            (1.0 - accepted_sparse_validity) * scale

        feature = coarse_feature
        for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
            feature, \
            scaleless_depth, \
            scale, \
            propagation_logits, \
            rejection_logits, \
            keep_mask = decoder_block(
                feature=feature,
                skip=features[level],
                scaleless_depth=scaleless_depth,
                scale=scale,
                sparse_depth=sparse_depth,
                propagation_logits=propagation_logits,
                rejection_logits=rejection_logits)

        # Apply scale to the scaleless depth before final output.
        depth = scaleless_depth * scale
        depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return depth
