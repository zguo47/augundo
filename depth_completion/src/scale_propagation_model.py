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


def sparse_depth_at_resolution(sparse_depth, output_size, return_indices=False):
    '''Sparse depth at requested resolution.
    '''

    # Pool only the positive-entries to locate a valid measurement.
    has_measurement, measurement_indices = functional.adaptive_max_pool2d(
        (sparse_depth > 0.0).to(sparse_depth.dtype),
        output_size=output_size,
        return_indices=True)
    selected_depth = torch.gather(
        sparse_depth.flatten(2),
        dim=2,
        index=measurement_indices.flatten(2))
    selected_depth = selected_depth.reshape_as(has_measurement) * has_measurement
    # Debug comparisons use the selected original pixel, not pooled ground truth.
    if return_indices:
        return selected_depth, measurement_indices
    return selected_depth


def normalize_rejection_input(feature, depth, sparse_depth):
    n_channel = feature.shape[1]
    feature = functional.layer_norm(
        feature.permute(0, 2, 3, 1),
        normalized_shape=(n_channel,)).permute(0, 3, 1, 2)
    return torch.cat([feature, depth, sparse_depth], dim=1)


def propagate_scale(
        scale,
        scaleless_depth,
        sparse_depth,
        keep_mask,
        propagation_logits,
        n_propagation_iterations=1):
    '''Inject accepted scale estimates once, then relay the persistent scale map.

    Arg(s):
        scale : torch.Tensor[float32]
            B x 1 x H x W inherited and learned positive scale map
        scaleless_depth : torch.Tensor[float32]
            B x 1 x H x W positive geometry; detached on the metric path
        sparse_depth : torch.Tensor[float32]
            B x 1 x H x W measurements sampled at this resolution
        keep_mask : torch.Tensor[float32]
            B x 1 x H x W keep decisions, zero at missing or rejected points
        propagation_logits : torch.Tensor[float32]
            B x 9 x H x W outgoing weights, indexed at the source pixel
        n_propagation_iterations : int
            number of neighboring exchanges at this resolution
    Returns:
            torch.Tensor[float32] : B x 1 x H x W refined scale map

    '''

    n_batch, _, n_height, n_width = scale.shape
    sparse_scale = sparse_depth / scaleless_depth.detach().clamp_min(1e-7)
    # Accepted measurements seed absolute scale, not a correction around 1.
    # Injection occurs only here. Neither iterations nor the return restore it.
    scale = keep_mask * sparse_scale + (1.0 - keep_mask) * scale
    outgoing_weights = torch.sigmoid(propagation_logits)

    # Each column of fold is one source pixel's 3 x 3 outgoing message patch.
    # Overlapping patches sum at their destinations. 
    incoming_weight = functional.fold(
        outgoing_weights.reshape(n_batch, 9, n_height * n_width),
        output_size=(n_height, n_width),
        kernel_size=3,
        padding=1)
    for _ in range(n_propagation_iterations):
        outgoing_scale = outgoing_weights * scale
        incoming_scale = functional.fold(
            outgoing_scale.reshape(n_batch, 9, n_height * n_width),
            output_size=(n_height, n_width),
            kernel_size=3,
            padding=1)

        # Average the current scale with incoming messages. The learned center
        # connection is retained in addition to this fixed self-retention term.
        scale = (scale + incoming_scale) / (1.0 + incoming_weight)
    return scale


class ScalePropagation(nn.Module):
    '''Expose propagation as a module so optional debug hooks see its inputs.'''

    def __init__(self, n_propagation_iterations):
        super(ScalePropagation, self).__init__()
        self.n_propagation_iterations = n_propagation_iterations

    def forward(self, scale, scaleless_depth, sparse_depth, keep_mask, logits):
        return propagate_scale(
            scale, scaleless_depth, sparse_depth, keep_mask, logits,
            self.n_propagation_iterations)


def normalize_scaleless_depth(depth):
    '''Fix the geometry's global magnitude; the scale map carries metric units.'''
    return depth / depth.mean(dim=[2, 3], keepdim=True).clamp_min(1e-7)


def rejection_keep_mask(rejection_logits, sparse_depth, temperature, training):
    '''Make keep/reject decisions for this level's sparse depth.
    '''

    if training:
        rejection_decision = functional.gumbel_softmax(
            rejection_logits,
            tau=temperature,
            hard=True,
            dim=1)
        keep_mask = rejection_decision[:, 0:1, :, :]
    else:
        rejection_decision = torch.argmax(
            rejection_logits,
            dim=1,
            keepdim=True)
        keep_mask = (rejection_decision == 0).to(sparse_depth.dtype)

    return keep_mask * (sparse_depth > 0.0).to(sparse_depth.dtype)


class PropagationDecoderBlock(nn.Module):
    '''Refine scaleless geometry and the inherited scale map at a finer level.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection,
            n_propagation_iterations=2,
            max_predict_depth=8.0):
        super(PropagationDecoderBlock, self).__init__()

        # Carry features, scaleless geometry, and scale separately. Add this
        # level's encoder skip and sparse measurements: C + C + 1 + 1 + 1.
        self.fusion = ConvolutionBlock(
            2 * n_channels + 3,
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.n_propagation_iterations = n_propagation_iterations
        self.max_predict_depth = max_predict_depth
        self.depth_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        # Start by retaining the inherited scale rather than changing it.
        nn.init.zeros_(self.scale_update.conv.weight)
        self.scale_propagation = ScalePropagation(n_propagation_iterations)

        # Heads see features, current metric depth, and current sparse depth.
        # Both heads are freshly evaluated at each level
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

    def forward(self, feature, skip, scaleless_depth, scale, sparse_depth):

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
        sparse_depth = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=skip.shape[-2:])

        feature = self.fusion(torch.cat([
            feature,
            skip,
            # Shared features can train under both objectives, but map inputs
            # are context only: SI losses must not train preceding scale heads.
            scaleless_depth.detach(),
            scale.detach(),
            sparse_depth
        ], dim=1))
        # Geometry stays positive and mean-one at every resolution. Only the
        # SI objectives directly train this prediction and its preceding maps.
        scaleless_depth = normalize_scaleless_depth(
            scaleless_depth * torch.exp(torch.tanh(self.depth_update(feature))))
        scale = scale * torch.exp(torch.tanh(self.scale_update(feature)))
        # Metric heads see current metric context without a gradient into Z.
        depth = scaleless_depth.detach() * scale

        propagation_input = torch.cat([feature, depth, sparse_depth], dim=1)
        if self.use_outlier_rejection:
            rejection_input = normalize_rejection_input(
                feature=feature,
                depth=depth / self.max_predict_depth,
                sparse_depth=sparse_depth / self.max_predict_depth)
            rejection_logits = \
                self.rejection_map(rejection_input) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                sparse_depth=sparse_depth,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            rejection_logits = torch.zeros_like(sparse_depth).repeat(1, 2, 1, 1)
            keep_mask = (sparse_depth > 0.0).to(sparse_depth.dtype)

        propagation_logits = self.propagation_map(propagation_input)
        scale = self.scale_propagation(
            scale, scaleless_depth, sparse_depth, keep_mask, propagation_logits)

        # No exact measurement restoration: the next decoder receives learned,
        # propagated scale, and makes its own fresh rejection decisions.
        return feature, scaleless_depth, scale, propagation_logits, rejection_logits, keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder with persistent geometry and metric scale at five levels.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0,
                 use_outlier_rejection=True,
                 n_propagation_iterations=(1, 2, 3, 4, 6)):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        # Iteration counts are ordered from the bottom to full resolution.
        # Coarse scales are denser; finer scales receive more exchanges.
        self.n_propagation_iterations = n_propagation_iterations

        # Takes in RGB and sparse depth together and produce 5 feature maps.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # Positive bottom prediction supplies both a normalized scaleless depth
        self.coarse_scaleless_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        nn.init.zeros_(self.scale_update.conv.weight)
        self.scale_propagation = ScalePropagation(n_propagation_iterations[0])
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))
        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=gumbel_temperature,
                use_outlier_rejection=use_outlier_rejection,
                n_propagation_iterations=n_iteration,
                max_predict_depth=max_predict_depth)
            for n_iteration in n_propagation_iterations[1:]
        ])

    def forward(self, image, sparse_depth, return_all_outputs=False):
        features = self.encoder(image, sparse_depth)
        coarse_feature = features[-1]
        coarse_sparse_depth = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=coarse_feature.shape[-2:])

        coarse_depth = functional.softplus(
            self.coarse_scaleless_depth(coarse_feature)) + 1e-3
        fallback_scale = torch.mean(coarse_depth, dim=[2, 3], keepdim=True)
        scaleless_depth = normalize_scaleless_depth(coarse_depth)
        metric_geometry = scaleless_depth.detach()
        propagation_input = torch.cat([
            coarse_feature,
            metric_geometry,
            coarse_sparse_depth
        ], dim=1)

        # Reject before computing either the global metric initialization or
        # local messages.
        if self.use_outlier_rejection:
            rejection_input = normalize_rejection_input(
                feature=coarse_feature,
                depth=metric_geometry,
                sparse_depth=coarse_sparse_depth / self.max_predict_depth)
            rejection_logits = \
                self.rejection_map(rejection_input) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                sparse_depth=coarse_sparse_depth,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            keep_mask = (coarse_sparse_depth > 0.0).to(sparse_depth.dtype)

        # Each accepted measurement proposes a metric scale d_sparse / scaleless.
        sparse_scale = coarse_sparse_depth / metric_geometry.clamp_min(1e-3)
        scale = (
            torch.sum(keep_mask * sparse_scale, dim=[2, 3], keepdim=True) +
            fallback_scale.detach()) / (
                torch.sum(keep_mask, dim=[2, 3], keepdim=True) + 1.0)
        # Initialize a dense constant scale from the existing global estimate,
        # then allow the head and sparse-guided propagation to refine it.
        scale = scale.expand_as(scaleless_depth)
        scale = scale * torch.exp(torch.tanh(self.scale_update(coarse_feature)))
        propagation_logits = self.propagation_map(propagation_input)
        scale = self.scale_propagation(
            scale, metric_geometry, coarse_sparse_depth, keep_mask, propagation_logits)

        feature = coarse_feature
        scaleless_depths = [scaleless_depth]
        scale_maps = [scale]
        for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
            feature, scaleless_depth, scale, _, _, _ = decoder_block(
                feature,
                features[level],
                scaleless_depth,
                scale,
                sparse_depth)
            scaleless_depths.append(scaleless_depth)
            scale_maps.append(scale)

        # Clamp before output.
        depth = torch.clamp(
            scaleless_depth * scale,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        if not return_all_outputs:
            return depth
        # Full resolution first. Explicit outputs also work with DataParallel;
        # no retained per-forward graph or replica-local cache is needed.
        scaleless_depths = scaleless_depths[::-1]
        scale_maps = scale_maps[::-1]
        return dict(
            depth=depth,
            scaleless_depths=scaleless_depths,
            scale_maps=scale_maps,
            metric_depths_for_loss=[
                geometry.detach() * metric_scale
                for geometry, metric_scale in zip(
                    scaleless_depths[:4], scale_maps[:4])])
