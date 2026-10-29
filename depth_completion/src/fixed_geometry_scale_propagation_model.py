import torch
import torch.nn as nn
import torch.nn.functional as functional

from scale_propagation_model import (
    Conv2d, ConvolutionBlock, ScaleUpdateBlock, ScalePropagation,
    ScalePropagationDepthModel, normalize_scaleless_depth,
    sparse_depth_at_resolution, measurement_rejection_input, rejection_keep_mask,
    refine_scale)


class GeometryAwarePropagationDecoderBlock(nn.Module):
    '''Refine scaleless geometry and the inherited scale map at a finer level.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection,
            n_propagation_iterations=4,
            max_predict_depth=8.0,
            n_scale_refinement=1,
            use_sparse_depth_features=False):
        super(GeometryAwarePropagationDecoderBlock, self).__init__()

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
        self.scale_update.pre_propagation_refinement = True
        self.pre_propagation_refinement = True
        # A separate head repairs residual errors AFTER sparse propagation.
        # It uses the original refinement inputs, not geometry-change inputs.
        self.post_scale_update = ScaleUpdateBlock(n_channels)
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
        previous_geometry = scaleless_depth.detach()
        # SI directly trains scaleless
        scaleless_depth = normalize_scaleless_depth(
            scaleless_depth * torch.exp(torch.tanh(self.depth_update(feature))))
        # Adapt the inherited scale to the UPDATED geometry before propagation.
        # C+5 channels: features, new Z, scale, old Z, current metric depth,
        # and log(new Z / old Z). Geometry inputs remain detached.
        # Zero-initialized output weights make the initial adaptation identity.
        geometry = scaleless_depth.detach()
        geometry_change = torch.log(geometry.clamp_min(1e-7)) - \
            torch.log(previous_geometry.clamp_min(1e-7))
        for _ in range(self.n_scale_refinement):
            adaptation_input = torch.cat([
                feature, geometry, scale, previous_geometry,
                geometry * scale, geometry_change
            ], dim=1)
            scale = scale * torch.exp(torch.tanh(self.scale_update(adaptation_input)))
        depth = geometry * scale

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

        # Adaptation and refinement have separate parameters and purposes.
        # Keep geometry detached inside refinement and do not restore anchors.
        scale = refine_scale(
            feature, scaleless_depth, scale, sparse_scale, keep_mask,
            self.post_scale_update,
            self.n_scale_refinement)
        return feature, scaleless_depth, scale, propagation_logits, rejection_logits, keep_mask



class FixedGeometryScalePropagationDepthModel(ScalePropagationDepthModel):
    '''Joint geometry/scale decoder replacing the fixed-geometry experiment.

    Keep the historical model name for existing repository entry points.
    Bottom initialization is unchanged from the original scale model. At finer
    levels, adapt inherited scale immediately after updating geometry, then
    propagate and refine the result with a separate head. Metric loss retains
    the original geometry-detachment option.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 encoder_type='conv',
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 training_stage='joint',
                 propagation_mode='hierarchical',
                 detach_scaleless_depth=True,
                 use_outlier_rejection=False,
                 n_scale_refinement=1,
                 propagation_anchor_strength=2.0,
                 use_sparse_depth_features=False):
        if training_stage != 'joint' or propagation_mode != 'hierarchical':
            raise ValueError('This replacement uses joint hierarchical training; '
                             'former geometry/scale stages and global mode are unsupported.')
        super(FixedGeometryScalePropagationDepthModel, self).__init__(
            min_predict_depth=min_predict_depth, max_predict_depth=max_predict_depth,
            n_channels=n_channels, n_head=n_head, encoder_type=encoder_type,
            n_propagation_iterations=n_propagation_iterations,
            detach_scaleless_depth=detach_scaleless_depth,
            use_outlier_rejection=use_outlier_rejection,
            n_scale_refinement=n_scale_refinement,
            propagation_anchor_strength=propagation_anchor_strength,
            use_sparse_depth_features=use_sparse_depth_features)
        self.training_stage = training_stage
        self.fixed_geometry = False
        self.pre_propagation_refinement = True
        self.post_propagation_refinement = True
        # Preserve initialization of shared layers for comparison with the old
        # model. Scale-update input width is unchanged; its five scalar inputs
        # now describe geometry change instead of post-propagation anchor error.
        for index, original in enumerate(self.decoder):
            block = GeometryAwarePropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=self.gumbel_temperature,
                use_outlier_rejection=use_outlier_rejection,
                n_propagation_iterations=n_propagation_iterations[index + 1],
                max_predict_depth=max_predict_depth,
                n_scale_refinement=n_scale_refinement,
                use_sparse_depth_features=use_sparse_depth_features)
            state = block.state_dict()
            state.update(original.state_dict())
            block.load_state_dict(state)
            self.decoder[index] = block
            block.scale_propagation.propagation_anchor_strength = propagation_anchor_strength
