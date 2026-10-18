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
    SparseScaleInjection,
    sparse_depth_to_points,
    features_at_points,
    point_scale_observations,
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

    def test_all_original_points_survive_coarse_cells_and_unequal_batch_counts(self):
        sparse = torch.zeros(3, 1, 5, 7)
        sparse[0, 0, 1, 1] = 10.
        sparse[0, 0, 1, 2] = 60.
        sparse[2, 0, 4, 6] = 7.
        points = sparse_depth_to_points(sparse)
        self.assertEqual(points['depth'].tolist(), [10., 60., 7.])
        self.assertEqual(points['batch'].tolist(), [0, 0, 2])
        self.assertEqual(points['x'].tolist(), [1, 2, 6])
        self.assertEqual(points['y'].tolist(), [1, 1, 4])
        geometry = torch.arange(1., 19.).reshape(3, 1, 2, 3)
        sampled = features_at_points(geometry, points)[:, 0]
        full = functional.interpolate(geometry, size=(5, 7), mode='bilinear', align_corners=True)
        expected = full[points['batch'], 0, points['y'], points['x']]
        torch.testing.assert_close(sampled, expected)

    def test_per_point_rejection_distinguishes_two_measurements_in_one_coarse_cell(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        with torch.no_grad():
            model.rejection_map.conv.weight.zero_()
            model.rejection_map.conv.weight[0, -1, 0, 0] = -4.
            model.rejection_bias.copy_(torch.tensor([1., 0.]).reshape(1, 2, 1, 1))
        sparse = torch.zeros(1, 1, 4, 4)
        sparse[0, 0, 1, 1], sparse[0, 0, 1, 2] = 10., 60.
        points = sparse_depth_to_points(sparse)
        ratio, keep, logits = point_scale_observations(
            torch.zeros(1, 8, 1, 1), torch.ones(1, 1, 1, 1), None, points,
            model.rejection_map, model.rejection_bias, True, 1., False, 100.)
        self.assertEqual(keep.tolist(), [1., 0.])
        self.assertEqual(ratio.tolist(), [10., 60.])
        self.assertEqual(logits.shape, (1, 2, 1, 2))

    def test_fine_pixels_can_choose_different_sources_from_the_same_coarse_region(self):
        injection = SparseScaleInjection(n_channels=4)
        with torch.no_grad():
            injection.query_projection.conv.weight.zero_()
            injection.source_projection.conv.weight.zero_()
            injection.query_projection.conv.weight[0, 0, 0, 0] = 10.
            injection.source_projection.conv.weight[0, 0, 0, 0] = 10.
        sparse = torch.zeros(1, 1, 4, 4)
        sparse[0, 0, 1, 1], sparse[0, 0, 1, 2] = 10., 60.
        points = sparse_depth_to_points(sparse)
        feature = torch.zeros(1, 4, 4, 4)
        feature[:, 0, :, :2], feature[:, 1, :, :2] = 1., -1.
        feature[:, 0, :, 2:], feature[:, 1, :, 2:] = -1., 1.
        geometry = torch.ones(1, 1, 4, 4)
        geometry[:, :, :, 2:] = 2.
        targets = points['depth'] / features_at_points(geometry, points)[:, 0]
        self.assertEqual(targets.tolist(), [10., 30.])
        # Both points remain present even when the preceding map is one cell.
        coarse, _ = injection(torch.full((1, 1, 1, 1), 2.), feature[:, :, :1, :1],
                              feature, torch.ones(1, 1, 1, 1), points, targets,
                              torch.ones_like(targets))
        result, weights = injection(torch.full_like(geometry, 2.), feature, feature,
                                    geometry, points, targets, torch.ones_like(targets))
        self.assertEqual(points['depth'].numel(), 2)
        self.assertEqual(coarse.shape, (1, 1, 1, 1))
        self.assertAlmostEqual(float(result[0, 0, 1, 1]), 6., places=5)
        self.assertAlmostEqual(float(result[0, 0, 1, 2]), 16., places=5)
        self.assertGreater(float(weights.sum()), 0.)

    def test_point_injection_is_local_batch_isolated_and_excludes_rejected_sources(self):
        injection = SparseScaleInjection(n_channels=4)
        sparse = torch.zeros(2, 1, 5, 7)
        sparse[0, 0, 0, 0] = 6.
        points = sparse_depth_to_points(sparse)
        feature, geometry = torch.zeros(2, 4, 5, 7), torch.ones(2, 1, 5, 7)
        scale = torch.full_like(geometry, 2.)
        result, weights = injection(scale, feature, feature, geometry, points,
                                    points['depth'], torch.ones(1))
        self.assertGreater(float(result[0, 0, 0, 0]), 2.)
        torch.testing.assert_close(result[1], scale[1])
        torch.testing.assert_close(result[0, :, 2:], scale[0, :, 2:])
        torch.testing.assert_close(result[0, :, :, 2:], scale[0, :, :, 2:])
        rejected, _ = injection(scale, feature, feature, geometry, points,
                                points['depth'], torch.zeros(1))
        torch.testing.assert_close(rejected, scale)

    def test_injection_and_observations_train_sources_queries_and_keep_not_geometry(self):
        torch.manual_seed(8)
        injection = SparseScaleInjection(n_channels=8)
        source = torch.randn(2, 8, 5, 7, requires_grad=True)
        feature = torch.randn(2, 8, 3, 4, requires_grad=True)
        geometry = (torch.rand(2, 1, 3, 4) + 0.5).requires_grad_()
        scale = (torch.rand_like(geometry) + 1.).requires_grad_()
        sparse = torch.zeros(2, 1, 5, 7)
        sparse[0, 0, 1, 2], sparse[1, 0, 3, 4] = 6., 8.
        points = sparse_depth_to_points(sparse)
        targets = points['depth'] / features_at_points(geometry.detach(), points)[:, 0]
        keep = torch.ones_like(targets, requires_grad=True)
        result, _ = injection(scale, feature, source, geometry, points, targets, keep)
        result.square().mean().backward()
        self.assertIsNone(geometry.grad)
        for value in (feature, source, scale, keep):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        for parameter in injection.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_every_level_reuses_all_points_and_global_initialization_counts_them_once(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        observed, raw_bottom, relays = [], [], []
        handles = [model.coarse_scaleless_depth.register_forward_hook(
            lambda module, inputs, output: raw_bottom.append(output))]
        for stage in [model] + list(model.decoder):
            handles.append(stage.sparse_injection.register_forward_pre_hook(
                lambda module, inputs: observed.append(inputs)))
            handles.append(stage.scale_propagation.register_forward_pre_hook(
                lambda module, inputs: relays.append(module.n_propagation_iterations)))
        sparse = torch.zeros(2, 1, 17, 25)
        sparse[0, 0, 1, 1], sparse[0, 0, 1, 2], sparse[1, 0, 16, 24] = 5., 9., 7.
        # No model constraint path may reduce the source list with grid pooling.
        with mock.patch('scale_propagation_model.sparse_depth_at_resolution', side_effect=AssertionError):
            with torch.no_grad():
                model(torch.rand(2, 3, 17, 25), sparse)
        for handle in handles:
            handle.remove()
        self.assertEqual(relays, [1, 2, 3, 4, 6])
        self.assertEqual(len(observed), 5)
        for level, inputs in enumerate(observed):
            scale, feature, source, geometry, points, ratio, keep = inputs
            self.assertIs(points, observed[0][4])
            self.assertEqual(ratio.numel(), 3)
            self.assertEqual(keep.tolist(), [1., 1., 1.])
            full = functional.interpolate(geometry.detach(), size=(17, 25),
                                          mode='bilinear', align_corners=True)
            matched = full[points['batch'], 0, points['y'], points['x']]
            torch.testing.assert_close(ratio, points['depth'] / matched.clamp_min(
                1e-3 if level == 0 else 1e-7))
        scale, _, _, _, points, ratio, keep = observed[0]
        fallback = (functional.softplus(raw_bottom[0]) + 1e-3).mean(dim=[2, 3], keepdim=True)
        expected = []
        for batch in range(2):
            selected = points['batch'] == batch
            expected.append((ratio[selected].sum() + fallback[batch, 0, 0, 0]) / (selected.sum() + 1.))
        torch.testing.assert_close(scale, torch.stack(expected)[:, None, None, None].expand_as(scale))

    def test_dense_relay_sends_scale_without_reinjecting_sources(self):
        scale = torch.full((1, 1, 5, 7), 2.)
        scale[0, 0, 2, 2] = 6.
        logits = torch.full((1, 9, 5, 7), -100.)
        logits[0, 5, 2, 2] = 100.
        logits[0, 5, 2, 3] = 100.
        first = propagate_scale(scale, logits, 1)
        second = propagate_scale(scale, logits, 2)
        self.assertEqual(float(first[0, 0, 2, 3]), 4.)
        self.assertEqual(float(first[0, 0, 2, 4]), 2.)
        self.assertEqual(float(second[0, 0, 2, 4]), 3.)

    def test_dense_relay_preserves_constants_and_does_not_wrap(self):
        scale = torch.full((2, 1, 3, 5), 2.)
        torch.testing.assert_close(propagate_scale(scale, torch.randn(2, 9, 3, 5), 6), scale)
        scale[0, 0, 0, 0] = 6.
        logits = torch.full((2, 9, 3, 5), -100.)
        logits[0, 0, 0, 0] = 100.
        torch.testing.assert_close(propagate_scale(scale, logits, 4), scale)

    def test_dense_relay_trains_weights_and_scale_and_does_not_restore_anchors(self):
        scale = torch.full((1, 1, 5, 5), 2.)
        scale[0, 0, 2, 2] = 6.
        scale.requires_grad_()
        logits = torch.zeros(1, 9, 5, 5, requires_grad=True)
        result = propagate_scale(scale, logits, 1)
        self.assertAlmostEqual(float(result[0, 0, 2, 2]), 17. / 5.5, places=6)
        result.square().mean().backward()
        for value in (scale, logits):
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

    def test_rejection_uses_each_points_own_measurement_and_normalized_context(self):
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
        sparse[0, 0, 7, 11], sparse[0, 0, 7, 12] = 55., 25.
        points = sparse_depth_to_points(sparse)
        with torch.no_grad():
            model(torch.rand(1, 3, 17, 25), sparse)
        for level, pair in enumerate(captured):
            rejection, propagation = pair['rejection'][0], pair['propagation'][0]
            torch.testing.assert_close(rejection[:, :8].mean(dim=1),
                                       torch.zeros_like(rejection[:, 0]), atol=1e-5, rtol=0.)
            torch.testing.assert_close(rejection[:, -1:].reshape(-1), points['depth'] / 100.)
            context = features_at_points(propagation[:, -1:], points)[:, 0]
            torch.testing.assert_close(rejection[:, -2:-1].reshape(-1),
                                       context / (1. if level == 0 else 100.))

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
            for name in ['scale_update', 'propagation_map', 'rejection_map', 'sparse_injection']:
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
