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
    '''Spatial convolutions (three by default); the first optionally downsamples.'''

    def __init__(self, in_channels, out_channels, stride, n_convolution=3):
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
        # Keep the existing three layers and their parameter names unchanged.
        for index in range(3, n_convolution):
            self.convolutions.add_module(str(index), Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)))

    def forward(self, feature):
        return self.convolutions(feature)


class SharedEncoder(nn.Module):
    '''Five-resolution encoder, with optional raw sparse-depth features.'''

    def __init__(self, n_channels, n_head, encoder_type='conv',
                 use_sparse_depth_features=False):
        super(SharedEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.use_sparse_depth_features = use_sparse_depth_features
        n_convolution = (3, 6, 6, 4, 4) if encoder_type == 'conv_attention' else (3, 3, 3, 3, 3)
        self.level0 = ConvolutionBlock(3 + int(use_sparse_depth_features), n_channels,
                                       stride=1, n_convolution=n_convolution[0])
        self.level1 = ConvolutionBlock(n_channels, n_channels, stride=2, n_convolution=n_convolution[1])
        self.level2 = ConvolutionBlock(n_channels, n_channels, stride=2, n_convolution=n_convolution[2])
        self.level3 = ConvolutionBlock(n_channels, n_channels, stride=2, n_convolution=n_convolution[3])
        self.level4 = ConvolutionBlock(n_channels, n_channels, stride=2, n_convolution=n_convolution[4])
        if encoder_type == 'conv_attention':
            # Reuse the existing full self-attention blocks at 1/8 and 1/16.
            self.level3_attention = nn.ModuleList([
                SelfAttentionBlock(n_channels=n_channels, n_head=n_head)
                for _ in range(2)
            ])
            self.level4_attention = nn.ModuleList([
                SelfAttentionBlock(n_channels=n_channels, n_head=n_head)
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

    def forward(self, image, sparse_depth=None):
        if self.use_sparse_depth_features:
            image = torch.cat([image, sparse_depth], dim=1)
        level0 = self.level0(image)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        if self.encoder_type == 'conv_attention':
            level3 = self.attention_level(level3, self.level3_attention)
        level4 = self.level4(level3)
        if self.encoder_type == 'conv_attention':
            level4 = self.attention_level(level4, self.level4_attention)

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


def feature_at_sparse_points(feature, measurement_indices, input_size):
    '''Sample a coarse map at the selected original-image measurement pixels.'''
    n_input_height, n_input_width = input_size
    # The sampler returns flattened indices into the original sparse image,
    # not coordinates of coarse-cell centers. Keep the per-image indices.
    y = (measurement_indices // n_input_width).to(feature.dtype)
    x = (measurement_indices % n_input_width).to(feature.dtype)
    y = 2.0 * y / max(n_input_height - 1, 1) - 1.0
    x = 2.0 * x / max(n_input_width - 1, 1) - 1.0
    grid = torch.cat([x, y], dim=1).permute(0, 2, 3, 1)

    return functional.grid_sample(
        feature,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)


def geometry_at_sparse_points(scaleless_depth, measurement_indices, input_size):
    '''Sample this level's detached geometry at selected original-image pixels.'''
    return feature_at_sparse_points(
        scaleless_depth.detach(), measurement_indices, input_size)


def measurement_rejection_input(
        feature, depth, sparse_depth, measurement_indices, input_size):

    n_input_height, n_input_width = input_size
    n_height, n_width = sparse_depth.shape[-2:]
    y = (measurement_indices // n_input_width).to(feature.dtype)
    x = (measurement_indices % n_input_width).to(feature.dtype)
    row = torch.arange(n_height, device=feature.device).reshape(1, 1, -1, 1)
    column = torch.arange(n_width, device=feature.device).reshape(1, 1, 1, -1)
    # Match adaptive_max_pool2d's bins, including odd input dimensions.
    y_start = row * n_input_height // n_height
    y_end = ((row + 1) * n_input_height + n_height - 1) // n_height
    x_start = column * n_input_width // n_width
    x_end = ((column + 1) * n_input_width + n_width - 1) // n_width
    measured = (sparse_depth > 0.0).to(feature.dtype)
    y_offset = (2.0 * (y - y_start + 0.5) / (y_end - y_start) - 1.0) * measured
    x_offset = (2.0 * (x - x_start + 0.5) / (x_end - x_start) - 1.0) * measured
    return torch.cat([
        normalize_rejection_input(feature, depth, sparse_depth),
        y_offset,
        x_offset
    ], dim=1)


def propagate_scale(
        scale,
        sparse_scale,
        keep_mask,
        propagation_logits,
        n_propagation_iterations=1):
    '''Inject accepted scale estimates once, then relay the persistent scale map.

    Arg(s):
        scale : torch.Tensor[float32]
            B x 1 x H x W inherited and learned positive scale map
        sparse_scale : torch.Tensor[float32]
            B x 1 x H x W measured depth divided by geometry at the same
            current-resolution grid pixel; computed with detached geometry
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
    # Accepting a measurement replaces the inherited scale at its location;
    # rejecting it leaves the inherited estimate available for propagation.
    # Inject only once, without restoring anchors inside or after the loop.
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
        incoming_scale = functional.fold(
            (outgoing_weights * scale).reshape(n_batch, 9, n_height * n_width),
            output_size=(n_height, n_width),
            kernel_size=3,
            padding=1)

        # Average the current scale with incoming scales. The denominator
        # depends on propagation weights.
        scale = (scale + incoming_scale) / (1.0 + incoming_weight)
    return scale


class ScalePropagation(nn.Module):
    '''Expose propagation as a module so optional debug hooks see its inputs.'''

    def __init__(self, n_propagation_iterations):
        super(ScalePropagation, self).__init__()
        self.n_propagation_iterations = n_propagation_iterations

    def forward(
            self, scale, scaleless_depth, sparse_depth, keep_mask, logits,
            sparse_scale):
        return propagate_scale(
            scale, sparse_scale, keep_mask, logits, self.n_propagation_iterations)


def normalize_scaleless_depth(depth):
    '''Fix the scaleless's global magnitude; the scale map carries metric units.'''
    return depth / depth.mean(dim=[2, 3], keepdim=True).clamp_min(1e-7)


def rejection_keep_mask(rejection_logits, sparse_depth, temperature, training):
    '''Make keep/reject decisions for this level's sparse depth.
    '''

    # Sample the same hard decisions during training and evaluation.
    rejection_decision = functional.gumbel_softmax(
        rejection_logits,
        tau=temperature,
        hard=True,
        dim=1)
    keep_mask = rejection_decision[:, 0:1, :, :]

    return keep_mask * (sparse_depth > 0.0).to(sparse_depth.dtype)

class ScaleUpdateBlock(nn.Module):
    '''Predict a nonlinear log-scale correction from the current refinement state.'''

    def __init__(self, n_channels):
        super(ScaleUpdateBlock, self).__init__()

        # C feature channels plus scaleless, current scale, accepted ratios,
        # keep mask, and the current discrepancy at accepted measurements.
        self.convolutions = nn.Sequential(
            Conv2d(
                n_channels + 5,
                n_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels,
                1,
                kernel_size=3,
                stride=1,
                activation_func=None))
        # Both passes initially preserve the propagated scale exactly.
        nn.init.zeros_(self.convolutions[-1].conv.weight)

    def forward(self, feature):
        return self.convolutions(feature)


def refine_scale(
        feature, scaleless_depth, scale, sparse_scale, keep_mask,
        scale_update, n_scale_refinement=1):
    '''Apply the configured weight-shared corrections after propagation, without restoration.'''
    for _ in range(n_scale_refinement):
        refinement_input = torch.cat([
            feature,
            scaleless_depth.detach(),
            scale,
            keep_mask * sparse_scale,
            keep_mask,
            keep_mask * (sparse_scale - scale)
        ], dim=1)
        scale = scale * torch.exp(torch.tanh(scale_update(refinement_input)))
    return scale


class PropagationDecoderBlock(nn.Module):
    '''Refine scaleless geometry and the inherited scale map at a finer level.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection,
            n_propagation_iterations=4,
            max_predict_depth=8.0,
            n_scale_refinement=2,
            use_sparse_depth_features=False):
        super(PropagationDecoderBlock, self).__init__()

        # Carry features, scaleless geometry, and scale separately. Add this
        # level's RGB encoder skip, without raw sparse depth: C + C + 1 + 1.
        # The optional sparse-feature mode adds one raw measurement channel.
        self.use_sparse_depth_features = use_sparse_depth_features
        self.fusion = ConvolutionBlock(
            2 * n_channels + 2 + int(use_sparse_depth_features),
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.n_propagation_iterations = n_propagation_iterations
        self.max_predict_depth = max_predict_depth
        self.n_scale_refinement = n_scale_refinement
        self.depth_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = ScaleUpdateBlock(n_channels)
        self.scale_propagation = ScalePropagation(n_propagation_iterations)

        # Heads see features, current metric depth, and current sparse depth.
        # Both heads are freshly evaluated at each level
        # Rejection additionally receives y/x offsets of the selected source.
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 4,
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
        input_size = sparse_depth.shape[-2:]
        sparse_depth, measurement_indices = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=skip.shape[-2:],
            return_indices=True)

        fusion_inputs = [
            feature,
            skip,
            scaleless_depth.detach(),
            scale.detach()
        ]
        if self.use_sparse_depth_features:
            fusion_inputs.append(sparse_depth)
        feature = self.fusion(torch.cat(fusion_inputs, dim=1))
        # SI directly trains scaleless
        scaleless_depth = normalize_scaleless_depth(
            scaleless_depth * torch.exp(torch.tanh(self.depth_update(feature))))
        # detach scaleless
        depth = scaleless_depth.detach() * scale

        propagation_input = torch.cat([feature, depth, sparse_depth], dim=1)
        if self.use_outlier_rejection:
            rejection_input = measurement_rejection_input(
                feature=feature,
                depth=depth / self.max_predict_depth,
                sparse_depth=sparse_depth / self.max_predict_depth,
                measurement_indices=measurement_indices,
                input_size=input_size)
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
        sparse_scale = sparse_depth / scaleless_depth.detach().clamp_min(1e-7)
        scale = self.scale_propagation(
            scale, scaleless_depth, sparse_depth, keep_mask, propagation_logits,
            sparse_scale)

        # Refine the actual propagated map, without restoring sparse anchors.
        scale = refine_scale(
            feature, scaleless_depth, scale, sparse_scale, keep_mask,
            self.scale_update,
            self.n_scale_refinement)
        # Keep the learned correction at all pixels, including accepted sources.
        return feature, scaleless_depth, scale, propagation_logits, rejection_logits, keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder with five levels.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0,
                 use_outlier_rejection=True,
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 detach_scaleless_depth=True,
                 n_scale_refinement=1,
                 encoder_type='conv',
                 use_sparse_depth_features=False):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.detach_scaleless_depth = detach_scaleless_depth
        self.n_scale_refinement = n_scale_refinement
        self.use_sparse_depth_features = use_sparse_depth_features
        # Iteration counts are ordered from the bottom to full resolution.
        # Coarse scales are denser; finer scales receive more exchanges.
        self.n_propagation_iterations = n_propagation_iterations

        # RGB-only by default; optionally concatenate raw sparse measurements.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head,
            encoder_type=encoder_type,
            use_sparse_depth_features=use_sparse_depth_features)

        # Positive bottom prediction supplies a normalized scaleless depth
        self.coarse_scaleless_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = ScaleUpdateBlock(n_channels)
        self.scale_propagation = ScalePropagation(n_propagation_iterations[0])
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 4,
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
                max_predict_depth=max_predict_depth,
                n_scale_refinement=n_scale_refinement,
                use_sparse_depth_features=use_sparse_depth_features)
            for n_iteration in n_propagation_iterations[1:]
        ])

    def forward(self, image, sparse_depth, return_all_outputs=False,
                return_rejection_outputs=False):
        features = self.encoder(image, sparse_depth)
        coarse_feature = features[-1]
        coarse_sparse_depth, measurement_indices = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=coarse_feature.shape[-2:],
            return_indices=True)

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

        # Reject before computing either the global scale initialization or
        # local scales.
        if self.use_outlier_rejection:
            # No sparse-derived scale exists until this decision is made. Use
            # the coarse prediction as the initial context;
            # normalize it and the measurement with the same dataset range.
            rejection_input = measurement_rejection_input(
                feature=coarse_feature,
                depth=coarse_depth.detach() / self.max_predict_depth,
                sparse_depth=coarse_sparse_depth / self.max_predict_depth,
                measurement_indices=measurement_indices,
                input_size=sparse_depth.shape[-2:])
            rejection_logits = \
                self.rejection_map(rejection_input) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                sparse_depth=coarse_sparse_depth,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            keep_mask = (coarse_sparse_depth > 0.0).to(sparse_depth.dtype)
            rejection_logits = torch.zeros_like(coarse_sparse_depth).repeat(1, 2, 1, 1)

        # Each accepted measurement proposes a metric scale d_sparse / scaleless.
        sparse_scale = coarse_sparse_depth / metric_geometry.clamp_min(1e-3)
        scale = (
            torch.sum(keep_mask * sparse_scale, dim=[2, 3], keepdim=True) +
            fallback_scale.detach()) / (
                torch.sum(keep_mask, dim=[2, 3], keepdim=True) + 1.0)
        # Initialize a dense constant scale from the existing global estimate,
        # then inject accepted sparse depth points and propagate the scale map.
        scale = scale.expand_as(scaleless_depth)
        propagation_logits = self.propagation_map(propagation_input)
        scale = self.scale_propagation(
            scale, metric_geometry, coarse_sparse_depth, keep_mask, propagation_logits,
            sparse_scale)
        scale = refine_scale(
            coarse_feature, scaleless_depth, scale, sparse_scale, keep_mask,
            self.scale_update,
            self.n_scale_refinement)

        feature = coarse_feature
        scaleless_depths = [scaleless_depth]
        scale_maps = [scale]

        rejection_levels = [rejection_logits] if return_rejection_outputs else []
        rejection_keep_masks = [keep_mask] if return_rejection_outputs else []
        for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
            feature, scaleless_depth, scale, _, rejection_logits, keep_mask = decoder_block(
                feature,
                features[level],
                scaleless_depth,
                scale,
                sparse_depth)
            scaleless_depths.append(scaleless_depth)
            scale_maps.append(scale)
            if return_rejection_outputs:
                rejection_levels.append(rejection_logits)
                rejection_keep_masks.append(keep_mask)

        # Clamp before output.
        depth = torch.clamp(
            scaleless_depth * scale,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        if not return_all_outputs and not return_rejection_outputs:
            return depth

        scaleless_depths = scaleless_depths[::-1]
        scale_maps = scale_maps[::-1]
        outputs = dict(
            depth=depth,
            scaleless_depths=scaleless_depths,
            scale_maps=scale_maps,
            metric_depths_for_loss=[
                (geometry.detach() if self.detach_scaleless_depth else geometry) * metric_scale
                for geometry, metric_scale in zip(
                    scaleless_depths, scale_maps)])
        if return_rejection_outputs:
            outputs['rejection_logits'] = rejection_levels[::-1]
            outputs['rejection_keep_masks'] = rejection_keep_masks[::-1]
        return outputs
