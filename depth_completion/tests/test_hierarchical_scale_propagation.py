import os
import sys
import unittest
from unittest import mock

from PIL import Image

import torch
import torch.nn.functional as functional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scale_propagation_model import (
    ScalePropagationDepthModel,
    geometry_at_sparse_points,
    normalize_rejection_input,
    propagate_scale,
    rejection_keep_mask,
    sparse_depth_at_resolution)
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from utils.src import loss_utils
from utils.src.transforms import Transforms


class HierarchicalScalePropagationTest(unittest.TestCase):

    def test_selects_first_valid_value_instead_of_averaging(self):
        sparse = torch.tensor([[[[2., 40., 0., 60.],
                                [70., 80., 90., 100.],
                                [0., 110., 6., 120.],
                                [130., 140., 150., 160.]]]])
        sampled = sparse_depth_at_resolution(sparse, (2, 2))
        self.assertTrue(torch.equal(
            sampled, torch.tensor([[[[2., 60.], [110., 6.]]]])))
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (4, 4)), sparse))

    def test_preserves_measurements_between_nearest_sampling_coordinates(self):
        sparse = torch.zeros(2, 1, 4, 4)
        sparse[0, 0, 1, 1] = 3.
        sparse[0, 0, 1, 3] = 5.
        sparse[0, 0, 3, 1] = 7.
        sparse[0, 0, 3, 3] = 9.
        expected = torch.tensor([[[[3., 5.], [7., 9.]]],
                                 [[[0., 0.], [0., 0.]]]])
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (2, 2)), expected))

    def test_selects_valid_measurements_for_odd_shapes(self):
        sparse = torch.zeros(1, 1, 5, 7)
        sparse[0, 0, 0, 1] = 3.
        sparse[0, 0, 4, 5] = 7.
        expected = torch.tensor([[[[3., 0., 0.], [0., 0., 7.]]]])
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (2, 3)), expected))

    def test_one_source_sends_absolute_scale_to_the_right(self):
        scale = torch.full((1, 1, 5, 5), 2.)
        geometry = torch.ones_like(scale) * 0.5
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 3.
        logits = torch.full((1, 9, 5, 5), -100.)
        logits[0, 5, 2, 2] = 100.
        result = propagate_scale(scale, sparse / geometry, (sparse > 0).float(), logits)
        expected = scale.clone()
        # Sparse scale is 3 / 0.5 = 6; the right pixel combines scales 2 and 6.
        expected[0, 0, 2, 2] = 6.
        expected[0, 0, 2, 3] = 4.
        torch.testing.assert_close(result, expected)

    def test_geometry_sampling_matches_upsampling_and_gathering_original_pixels(self):
        sparse = torch.zeros(2, 1, 5, 7)
        sparse[0, 0, 1, 1] = 6.
        sparse[0, 0, 4, 5] = 10.
        sparse[1, 0, 0, 2] = 12.
        sparse[1, 0, 3, 4] = 15.
        geometry = torch.arange(1., 13.).reshape(2, 1, 2, 3).requires_grad_()
        _, indices = sparse_depth_at_resolution(sparse, (2, 3), return_indices=True)
        sampled = geometry_at_sparse_points(geometry, indices, sparse.shape[-2:])
        full_geometry = functional.interpolate(
            geometry.detach(), size=sparse.shape[-2:], mode='bilinear', align_corners=True)
        expected = torch.gather(full_geometry.flatten(2), 2, indices.flatten(2)).reshape_as(sampled)
        torch.testing.assert_close(sampled, expected)
        self.assertFalse(sampled.requires_grad)

    def test_geometry_sampling_handles_full_resolution_and_single_pixel_axes(self):
        for input_size, level_size in [((3, 5), (3, 5)), ((1, 7), (1, 3)),
                                       ((7, 1), (3, 1)), ((1, 1), (1, 1))]:
            sparse = torch.ones(2, 1, *input_size)
            geometry = torch.rand(2, 1, *level_size) + 0.5
            _, indices = sparse_depth_at_resolution(sparse, level_size, return_indices=True)
            sampled = geometry_at_sparse_points(geometry, indices, input_size)
            full_geometry = functional.interpolate(
                geometry, size=input_size, mode='bilinear', align_corners=True)
            expected = torch.gather(full_geometry.flatten(2), 2, indices.flatten(2)).reshape_as(sampled)
            torch.testing.assert_close(sampled, expected)

    def test_aligned_scale_ratio_is_injected_directly_without_redivision(self):
        sparse = torch.zeros(1, 1, 4, 4)
        sparse[0, 0, 1, 1] = 6.
        geometry = torch.tensor([[[[1., 3.], [2., 4.]]]])
        selected, indices = sparse_depth_at_resolution(sparse, (2, 2), return_indices=True)
        sampled = geometry_at_sparse_points(geometry, indices, sparse.shape[-2:])
        # At original pixel (1, 1), interpolated Z is 2, not coarse-cell Z = 1.
        self.assertAlmostEqual(float(sampled[0, 0, 0, 0]), 2., places=6)
        target = selected / sampled.clamp_min(1e-7)
        scale = torch.full_like(geometry, 2.)
        logits = torch.full((1, 9, 2, 2), -100.)
        propagated = propagate_scale(scale, target, (selected > 0).float(), logits)
        expected = scale.clone()
        expected[0, 0, 0, 0] = 3.
        torch.testing.assert_close(propagated, expected)
        self.assertNotEqual(float(target[0, 0, 0, 0]), float((selected / geometry)[0, 0, 0, 0]))

    def test_every_level_uses_aligned_ratios_and_bottom_reuses_them_for_initialization(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        observed, raw_bottom = [], []
        handles = [model.coarse_scaleless_depth.register_forward_hook(
            lambda module, inputs, output: raw_bottom.append(output))]
        for stage in [model] + list(model.decoder):
            handles.append(stage.scale_propagation.register_forward_pre_hook(
                lambda module, inputs: observed.append((module.n_propagation_iterations, inputs))))
        sparse = torch.zeros(2, 1, 17, 25)
        sparse[0, 0, 7, 11] = 5.
        sparse[1, 0, 16, 24] = 7.
        with torch.no_grad():
            model(torch.rand(2, 3, 17, 25), sparse)
        for handle in handles:
            handle.remove()
        self.assertEqual([iteration for iteration, _ in observed], [1, 2, 3, 4, 6])
        for level, (_, inputs) in enumerate(observed):
            scale, geometry, selected, keep, logits, ratio = inputs
            expected_sparse, indices = sparse_depth_at_resolution(
                sparse, geometry.shape[-2:], return_indices=True)
            full_geometry = functional.interpolate(
                geometry.detach(), size=sparse.shape[-2:], mode='bilinear', align_corners=True)
            matched = torch.gather(full_geometry.flatten(2), 2, indices.flatten(2)).reshape_as(selected)
            torch.testing.assert_close(selected, expected_sparse)
            epsilon = 1e-3 if level == 0 else 1e-7
            torch.testing.assert_close(ratio, selected / matched.clamp_min(epsilon))
            self.assertEqual(ratio.shape, scale.shape)
        scale, _, _, keep, _, ratio = observed[0][1]
        fallback = (functional.softplus(raw_bottom[0]) + 1e-3).mean(dim=[2, 3], keepdim=True)
        expected = ((keep * ratio).sum(dim=[2, 3], keepdim=True) + fallback) / (
            keep.sum(dim=[2, 3], keepdim=True) + 1.)
        torch.testing.assert_close(scale, expected.expand_as(scale))
        self.assertFalse(any('surface_compatibility' in name for name in model.state_dict()))

    def test_scales_relay_beyond_the_original_source_neighborhood(self):
        scale = torch.full((1, 1, 5, 7), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float()
        logits = torch.full((1, 9, 5, 7), -100.)
        logits[0, 5, 2, 2] = 100.
        logits[0, 5, 2, 3] = 100.
        first = propagate_scale(scale, sparse, keep, logits, 1)
        second = propagate_scale(scale, sparse, keep, logits, 2)
        self.assertEqual(float(first[0, 0, 2, 4]), 2.)
        self.assertEqual(float(second[0, 0, 2, 4]), 3.)
        # A rejected source can relay scale, but cannot inject its measurement.
        sparse[0, 0, 2, 3] = 100.
        torch.testing.assert_close(
            propagate_scale(scale, sparse, keep, logits, 2), second)

    def test_measurements_are_not_restored_during_or_after_propagation(self):
        scale = torch.full((1, 1, 5, 5), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        logits = torch.zeros(1, 9, 5, 5)
        # Sigmoid(0) = 0.5. The center receives eight scales 2 and its own 6.
        # Its new scale is (6 + 0.5 * (8 * 2 + 6)) / (1 + 9 * 0.5).
        first = propagate_scale(
            scale, sparse, (sparse > 0).float(), logits, 1)
        self.assertAlmostEqual(float(first[0, 0, 2, 2]), 17. / 5.5, places=6)
        second = propagate_scale(
            scale, sparse, (sparse > 0).float(), logits, 2)
        incoming = functional.fold(
            (torch.sigmoid(logits) * first).reshape(1, 9, 25),
            (5, 5), kernel_size=3, padding=1)
        weight = functional.fold(torch.sigmoid(logits).reshape(1, 9, 25),
                                 (5, 5), kernel_size=3, padding=1)
        torch.testing.assert_close(second, (first + incoming) / (1 + weight))
        self.assertNotEqual(float(second[0, 0, 2, 2]), 6.)

    def test_no_messages_preserve_inherited_scale_except_once_injected_sources(self):
        scale = torch.rand(1, 1, 5, 7) + 1.
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        logits = torch.full((1, 9, 5, 7), -100.)
        expected = scale.clone()
        expected[0, 0, 2, 2] = 6.
        torch.testing.assert_close(
            propagate_scale(scale, sparse,
                            (sparse > 0).float(), logits, 6), expected)

    def test_constant_scale_stays_constant_with_empty_or_rejected_sources(self):
        scale = torch.full((2, 1, 5, 7), 3.)
        geometry = torch.rand_like(scale) + 0.1
        sparse = torch.full_like(scale, 50.)
        logits = torch.randn(2, 9, 5, 7)
        keep = torch.zeros_like(scale, requires_grad=True)
        result = propagate_scale(scale, sparse / geometry, keep, logits, 6)
        torch.testing.assert_close(result, scale)
        result.sum().backward()
        self.assertTrue(torch.isfinite(keep.grad).all())
        self.assertGreater(float(keep.grad.abs().sum()), 0.)

    def test_outgoing_boundary_messages_do_not_wrap(self):
        scale = torch.full((1, 1, 3, 3), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 0, 0] = 6.
        logits = torch.full((1, 9, 3, 3), -100.)
        logits[0, 0, 0, 0] = 100.
        expected = scale.clone()
        expected[0, 0, 0, 0] = 6.
        torch.testing.assert_close(
            propagate_scale(scale, sparse,
                            (sparse > 0).float(), logits, 4), expected)

    def test_propagation_trains_scale_weights_and_keep_but_not_geometry(self):
        scale = torch.full((1, 1, 5, 7), 2., requires_grad=True)
        geometry = torch.ones_like(scale, requires_grad=True)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float().requires_grad_()
        logits = torch.zeros(1, 9, 5, 7, requires_grad=True)
        _, indices = sparse_depth_at_resolution(sparse, scale.shape[-2:], return_indices=True)
        matched = geometry_at_sparse_points(geometry, indices, sparse.shape[-2:])
        result = propagate_scale(scale, sparse / matched.clamp_min(1e-7), keep, logits, 4)
        result.sum().backward()
        self.assertIsNone(geometry.grad)
        for value in (scale, keep, logits):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_rejection_input_normalizes_features_without_mutating_them(self):
        feature = torch.arange(8.).reshape(1, 8, 1, 1).expand(1, 8, 3, 4).clone()
        original = feature.clone()
        feature.requires_grad_()
        depth = torch.full((1, 1, 3, 4), 60.)
        sparse = torch.full_like(depth, 55.)
        result = normalize_rejection_input(feature, depth / 100., sparse / 100.)
        self.assertTrue(torch.equal(feature, original))
        torch.testing.assert_close(result[:, :8].mean(dim=1), torch.zeros(1, 3, 4))
        torch.testing.assert_close(result[:, :8].var(dim=1, unbiased=False), torch.ones(1, 3, 4))
        torch.testing.assert_close(result[:, -2:-1], torch.full_like(depth, 0.6))
        result[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_only_rejection_receives_scaled_depth_at_every_level(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        captured = []
        for stage in [model] + list(model.decoder):
            pair = dict(rejection=[], propagation=[])
            stage.rejection_map.register_forward_pre_hook(
                lambda module, inputs, pair=pair: pair['rejection'].append(inputs[0].clone()))
            stage.propagation_map.register_forward_pre_hook(
                lambda module, inputs, pair=pair: pair['propagation'].append(inputs[0].clone()))
            captured.append(pair)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[0, 0, 7, 11] = 55.
        with torch.no_grad():
            model(torch.rand(1, 3, 17, 25), sparse)
        for level, pair in enumerate(captured):
            rejection, propagation = pair['rejection'][0], pair['propagation'][0]
            torch.testing.assert_close(rejection[:, :8].mean(dim=1),
                                       torch.zeros_like(rejection[:, 0]), atol=1e-5, rtol=0.)
            torch.testing.assert_close(rejection[:, -1:], propagation[:, -1:] / 100.)
            torch.testing.assert_close(
                rejection[:, -2:-1], propagation[:, -2:-1] / (1. if level == 0 else 100.))

    def test_rejection_masks_missing_points_and_has_finite_gradients(self):
        sparse = torch.tensor([[[[0., 3.], [4., 0.]]]])
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        keep = rejection_keep_mask(logits, sparse, 1., training=True)
        self.assertTrue(torch.equal(keep[sparse == 0], torch.zeros(2)))
        self.assertTrue(((keep == 0) | (keep == 1)).all())
        keep.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def make_wrapper(self, rejection=False):
        wrapper = ScalePropagationDepthCompletionModel.__new__(ScalePropagationDepthCompletionModel)
        wrapper.min_predict_depth = 0.01
        wrapper.max_predict_depth = 100.
        wrapper.model_depth = ScalePropagationDepthModel(
            n_channels=8, min_predict_depth=0.01, max_predict_depth=100.,
            use_outlier_rejection=rejection)
        return wrapper

    def test_odd_shapes_empty_inputs_and_backward(self):
        torch.manual_seed(4)
        for rejection in (True, False):
            wrapper = self.make_wrapper(rejection)
            image = torch.rand(2, 3, 17, 25)
            sparse = torch.zeros(2, 1, 17, 25)
            sparse[1, 0, 7, 11] = 5.
            outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
            self.assertEqual(len(outputs), 16)
            self.assertTrue(all(value.shape == sparse.shape for value in outputs))
            self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
            target = torch.rand_like(sparse) * 5 + 1.
            loss, info = wrapper.compute_loss_supervised(target, outputs, {})
            loss.backward()
            gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(wrapper.model_depth.coarse_scaleless_depth.conv.weight.grad.abs().sum()), 0.)
            self.assertEqual(len([key for key in info if key.startswith('loss_si_r')]), 5)

    def test_fresh_rejection_at_each_level_and_deterministic_eval(self):
        model = ScalePropagationDepthModel(n_channels=8).eval()
        decisions = []
        with torch.no_grad():
            model.rejection_map.conv.weight.zero_()
            model.rejection_bias.copy_(torch.tensor([-100., 100.]).view(1, 2, 1, 1))
            for i, block in enumerate(model.decoder):
                block.rejection_map.conv.weight.zero_()
                bias = [100., -100.] if i % 2 == 0 else [-100., 100.]
                block.rejection_bias.copy_(torch.tensor(bias).view(1, 2, 1, 1))
                block.register_forward_hook(
                    lambda module, inputs, output: decisions.append(output[-1].clone()))
            image = torch.rand(1, 3, 17, 25)
            sparse = torch.ones(1, 1, 17, 25) * 5.
            first = model(image, sparse)
            second = model(image, sparse)
        self.assertTrue(torch.equal(first, second))
        for i, keep in enumerate(decisions[:4]):
            self.assertTrue(torch.equal(keep, torch.full_like(keep, 1. if i % 2 == 0 else 0.)))

    def test_constant_bottom_scale_is_carried_to_every_level_without_measurements(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        received = []
        model.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: received.append(inputs[0].clone()))
        with torch.no_grad():
            outputs = model(torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25), True)
        initial = received[0]
        torch.testing.assert_close(initial, initial[:, :, :1, :1].expand_as(initial))
        for geometry, scale in zip(outputs['scaleless_depths'], outputs['scale_maps']):
            torch.testing.assert_close(geometry.mean(dim=[2, 3]), torch.ones(1, 1))
            torch.testing.assert_close(scale, initial[:, :, :1, :1].expand_as(scale))
        self.assertEqual(model.n_propagation_iterations, (1, 2, 3, 4, 6))
        self.assertEqual([block.n_propagation_iterations for block in model.decoder], [2, 3, 4, 6])
        for stage in [model] + list(model.decoder):
            self.assertEqual(int(torch.count_nonzero(stage.scale_update.conv.weight)), 0)

    def test_scale_head_can_change_scale_without_sparse_measurements(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        image, sparse = torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25)
        with torch.no_grad():
            before = model(image, sparse, True)['scale_maps'][0]
            model.decoder[-1].scale_update.conv.weight.fill_(0.1)
            after = model(image, sparse, True)['scale_maps'][0]
        self.assertFalse(torch.equal(before, after))

    def test_metric_losses_do_not_train_geometry_heads(self):
        model = self.make_wrapper().model_depth
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 11] = 5.
        outputs = model(torch.rand(1, 3, 17, 25), sparse, True)
        sum(torch.log1p((value - 3.).abs()).mean()
            for value in outputs['metric_depths_for_loss']).backward()
        for stage, name in [(model, 'coarse_scaleless_depth')] + [
                (block, 'depth_update') for block in model.decoder]:
            self.assertIsNone(getattr(stage, name).conv.weight.grad)
        self.assertTrue(any(stage.scale_update.conv.weight.grad.abs().sum() > 0
                            for stage in [model] + list(model.decoder)))

    def test_si_losses_do_not_train_scale_propagation_or_rejection_heads(self):
        model = self.make_wrapper(True).model_depth
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, ::3, ::3] = 5.
        outputs = model(torch.rand(1, 3, 17, 25), sparse, True)
        loss = sum(loss_utils.scale_invariant_loss_func(
            value, torch.rand_like(value) + 1., torch.ones_like(value))
            for value in outputs['scaleless_depths'])
        loss.backward()
        for stage in [model] + list(model.decoder):
            for name in ['scale_update', 'propagation_map', 'rejection_map']:
                self.assertTrue(all(parameter.grad is None
                                    for parameter in getattr(stage, name).parameters()))
        self.assertGreater(float(model.coarse_scaleless_depth.conv.weight.grad.abs().sum()), 0.)

    def test_si_loss_is_scale_invariant_and_ignores_invalid_targets(self):
        geometry = torch.tensor([[[[1., 2.], [4., 8.]]],
                                 [[[2., 4.], [3., 5.]]]], requires_grad=True)
        target = torch.tensor([[[[2., 5.], [7., 0.]]],
                               [[[3., 4.], [9., 0.]]]])
        weights = (target > 0).float()
        loss = loss_utils.scale_invariant_loss_func(geometry, target, weights)
        scaled = geometry * torch.tensor([7., 0.2]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(
            loss, loss_utils.scale_invariant_loss_func(scaled, target, weights))
        loss.backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertTrue((geometry.grad[target == 0] == 0).all())

    def test_empty_target_has_finite_zero_loss(self):
        geometry = torch.ones(2, 1, 3, 4, requires_grad=True)
        loss = loss_utils.scale_invariant_loss_func(
            geometry, torch.zeros_like(geometry), torch.zeros_like(geometry))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue((geometry.grad == 0).all())

    def test_decoder_losses_have_the_requested_resolution_weights(self):
        wrapper = self.make_wrapper()
        target = torch.ones(1, 1, 3, 4) * 2.
        geometry = [torch.ones_like(target) for _ in range(5)]
        scale = [torch.ones_like(target) for _ in range(5)]
        metric = [torch.ones_like(target) * (3. + r) for r in range(5)]
        outputs = [target] + geometry + scale + metric
        loss, info = wrapper.compute_loss_supervised(target, outputs, {})
        expected = sum(torch.log(torch.tensor(2. + r)) / (4 ** r) for r in range(5))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(float(info['loss_si']), 0.)

    def test_bottom_losses_contribute_with_one_over_256_weight(self):
        wrapper = self.make_wrapper()
        target = torch.ones(1, 1, 2, 2) * 2.
        geometry = [torch.ones_like(target) for _ in range(4)] + [
            torch.tensor([[[[0.5, 1.], [1., 1.5]]]], requires_grad=True)]
        metric = [target.clone() for _ in range(4)] + [
            torch.full_like(target, 5., requires_grad=True)]
        outputs = [target] + geometry + [torch.ones_like(target)] * 5 + metric
        loss, info = wrapper.compute_loss_supervised(target, outputs, {})
        expected_si = loss_utils.scale_invariant_loss_func(
            geometry[-1], target, torch.ones_like(target))
        expected_metric = torch.log(torch.tensor(4.))
        torch.testing.assert_close(info['loss_si_r4'], expected_si)
        torch.testing.assert_close(info['loss_log_l1_r4'], expected_metric)
        torch.testing.assert_close(loss, (expected_si + expected_metric) / 256.)
        loss.backward()
        self.assertGreater(float(geometry[-1].grad.abs().sum()), 0.)
        self.assertGreater(float(metric[-1].grad.abs().sum()), 0.)

    def test_all_auxiliary_outputs_follow_geometric_reversal(self):
        wrapper = self.make_wrapper()
        image, sparse = torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25)
        outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        expected = [value.flip(-1).clone() for value in outputs]
        # Adapt only the test environment to this repository's older Pillow API.
        lanczos = getattr(Image, 'Resampling', Image).LANCZOS
        with mock.patch.object(Image, 'ANTIALIAS', lanczos, create=True):
            transforms = Transforms()
        reversed_outputs = transforms.reverse_transform(
            outputs, {'random_horizontal_flip': torch.tensor([True])})
        for original, aligned in zip(expected, reversed_outputs):
            torch.testing.assert_close(aligned, original)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
