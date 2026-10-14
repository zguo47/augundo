import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scale_propagation_model import (
    ScalePropagationDepthModel,
    normalize_rejection_input,
    propagate_scale,
    rejection_keep_mask,
    sparse_depth_at_resolution)


class HierarchicalScalePropagationTest(unittest.TestCase):

    def test_selects_first_valid_value_instead_of_averaging(self):
        sparse = torch.tensor([[[[2., 40., 0., 60.],
                                [70., 80., 90., 100.],
                                [0., 110., 6., 120.],
                                [130., 140., 150., 160.]]]])
        sampled = sparse_depth_at_resolution(sparse, (2, 2))
        self.assertTrue(torch.equal(
            sampled, torch.tensor([[[[2., 60.], [110., 6.]]]])))
        self.assertTrue(torch.equal(
            sparse_depth_at_resolution(sparse, (4, 4)), sparse))

    def test_preserves_measurements_between_nearest_sampling_coordinates(self):
        sparse = torch.zeros(2, 1, 4, 4)
        sparse[0, 0, 1, 1] = 3.
        sparse[0, 0, 1, 3] = 5.
        sparse[0, 0, 3, 1] = 7.
        sparse[0, 0, 3, 3] = 9.
        expected = torch.tensor([[[[3., 5.], [7., 9.]]],
                                 [[[0., 0.], [0., 0.]]]])
        self.assertTrue(torch.equal(
            sparse_depth_at_resolution(sparse, (2, 2)), expected))

    def test_selects_valid_measurements_for_odd_shapes(self):
        sparse = torch.zeros(1, 1, 5, 7)
        sparse[0, 0, 0, 1] = 3.
        sparse[0, 0, 4, 5] = 7.
        sampled = sparse_depth_at_resolution(sparse, (2, 3))
        expected = torch.tensor([[[[3., 0., 0.], [0., 0., 7.]]]])
        self.assertTrue(torch.equal(sampled, expected))

    def test_one_source_sends_to_the_right(self):
        depth = torch.full((1, 1, 5, 5), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float()
        logits = torch.full((1, 9, 5, 5), -100.)
        logits[0, 5, 2, 2] = 100.
        result = propagate_scale(depth, sparse, keep, logits)
        expected = depth.clone()
        expected[0, 0, 2, 2] = 6.
        # Source scale is 6 / 2 = 3. The destination combines that message
        # with its previous correction: 2 * (1 + 3) / (1 + 1) = 4.
        # The accepted source itself is restored exactly to its measurement.
        expected[0, 0, 2, 3] = 4.
        torch.testing.assert_close(result, expected)

    def test_corrections_relay_beyond_the_original_source_neighborhood(self):
        depth = torch.full((1, 1, 5, 7), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float()
        logits = torch.full((1, 9, 5, 7), -100.)
        logits[0, 5, 2, 2] = 100.
        logits[0, 5, 2, 3] = 100.
        first = propagate_scale(depth, sparse, keep, logits, 1)
        second = propagate_scale(depth, sparse, keep, logits, 2)
        self.assertEqual(float(first[0, 0, 2, 4]), 2.)
        self.assertEqual(float(second[0, 0, 2, 4]), 3.)
        self.assertEqual(float(second[0, 0, 2, 3]), 5.)
        self.assertEqual(float(second[0, 0, 2, 2]), 6.)

        # A rejected point cannot originate its own measurement, but it can
        # relay the accepted source's correction to the next pixel.
        sparse[0, 0, 2, 3] = 100.
        torch.testing.assert_close(
            propagate_scale(depth, sparse, keep, logits, 2), second)

    def test_propagates_scale_instead_of_averaging_metric_depth(self):
        depth = torch.full((1, 1, 3, 4), 2.)
        depth[0, 0, 1, 2] = 8.
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 1, 1] = 6.
        logits = torch.full((1, 9, 3, 4), -100.)
        logits[0, 5, 1, 1] = 100.
        result = propagate_scale(depth, sparse, (sparse > 0).float(), logits)
        # Neighbor keeps its base depth 8 and adopts correction (1 + 3) / 2.
        self.assertEqual(float(result[0, 0, 1, 2]), 16.)

    def test_restores_exact_anchors_and_trains_relay_weights(self):
        depth = torch.full((1, 1, 5, 7), 2., requires_grad=True)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        logits = torch.zeros(1, 9, 5, 7, requires_grad=True)
        keep = (sparse > 0).float()
        result = propagate_scale(depth, sparse, keep, logits, 4)
        self.assertTrue(torch.equal(result[keep > 0], sparse[keep > 0]))
        result[0, 0, 2, 4].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0, :, 2, 3].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_accepted_anchor_error_trains_its_keep_decision(self):
        depth = torch.full((1, 1, 5, 5), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        keep = torch.zeros_like(depth)
        keep[0, 0, 2, 2] = 1.
        keep.requires_grad_()
        logits = torch.zeros(1, 9, 5, 5)
        result = propagate_scale(depth, sparse, keep, logits, 8)
        self.assertEqual(float(result[0, 0, 2, 2]), 6.)
        # This measurement is wrong relative to target 3. Its own loss must
        # now provide a gradient that pushes the keep decision downward.
        loss = torch.log1p(torch.abs(result[0, 0, 2, 2] - 3.))
        loss.backward()
        self.assertTrue(torch.isfinite(keep.grad).all())
        self.assertGreater(float(keep.grad[0, 0, 2, 2]), 0.)

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
        torch.testing.assert_close(result[:, -1:], torch.full_like(sparse, 0.55))
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
            depth_scale = 1. if level == 0 else 100.
            torch.testing.assert_close(rejection[:, -2:-1], propagation[:, -2:-1] / depth_scale)

    def test_zero_gates_preserve_detail_except_at_accepted_anchors(self):
        depth = torch.rand(1, 1, 5, 7) + 1.
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        logits = torch.full((1, 9, 5, 7), -100.)
        expected = depth.clone()
        expected[0, 0, 2, 2] = 6.
        self.assertTrue(torch.equal(
            propagate_scale(depth, sparse, (sparse > 0).float(), logits, 16),
            expected))

    def test_outgoing_boundary_messages_do_not_wrap(self):
        depth = torch.full((1, 1, 3, 3), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 0, 0] = 6.
        logits = torch.full((1, 9, 3, 3), -100.)
        logits[0, 0, 0, 0] = 100.
        expected = depth.clone()
        expected[0, 0, 0, 0] = 6.
        torch.testing.assert_close(
            propagate_scale(depth, sparse, (sparse > 0).float(), logits, 4),
            expected)

    def test_rejected_and_empty_sources_preserve_dense_prediction(self):
        depth = torch.rand(2, 1, 5, 7) + 1.
        sparse = torch.full_like(depth, 50.)
        logits = torch.zeros(2, 9, 5, 7)
        keep = torch.zeros_like(depth, requires_grad=True)
        result = propagate_scale(depth, sparse, keep, logits, 16)
        self.assertTrue(torch.equal(result, depth))
        result.sum().backward()
        self.assertTrue(torch.isfinite(keep.grad).all())
        self.assertGreater(float(keep.grad.abs().sum()), 0.)
        sparse.zero_()
        self.assertTrue(torch.equal(
            propagate_scale(depth, sparse, torch.zeros_like(depth), logits, 16),
            depth))

    def test_rejection_masks_missing_points_and_has_finite_gradients(self):
        sparse = torch.tensor([[[[0., 3.], [4., 0.]]]])
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        keep = rejection_keep_mask(logits, sparse, 1., training=True)
        self.assertTrue(torch.equal(keep[sparse == 0], torch.zeros(2)))
        self.assertTrue(((keep == 0) | (keep == 1)).all())
        keep.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_odd_shapes_empty_inputs_and_backward(self):
        torch.manual_seed(4)
        for rejection in (True, False):
            model = ScalePropagationDepthModel(
                n_channels=8, min_predict_depth=0.01,
                max_predict_depth=100., use_outlier_rejection=rejection)
            image = torch.rand(2, 3, 17, 25)
            sparse = torch.zeros(2, 1, 17, 25)
            # The first image has no sparse points. The second has a point
            # between the coordinates that ordinary nearest sampling selects.
            sparse[1, 0, 7, 11] = 5.
            output = model(image, sparse)
            self.assertEqual(output.shape, sparse.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output > 0).all())
            output.mean().backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(model.coarse_scaleless_depth.conv.weight.grad.abs().sum()), 0.)

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
            self.assertTrue(torch.equal(
                keep, torch.full_like(keep, 1. if i % 2 == 0 else 0.)))

    def test_decoder_receives_and_returns_restored_metric_anchors(self):
        model = ScalePropagationDepthModel(
            n_channels=8, use_outlier_rejection=False).eval()
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[0, 0, 7, 11] = 5.
        received = []
        decoded = []
        model.decoder[0].register_forward_pre_hook(
            lambda module, inputs: received.append(inputs[2].clone()))
        for block in model.decoder:
            block.register_forward_hook(
                lambda module, inputs, output: decoded.append(output[1].clone()))
        with torch.no_grad():
            model(torch.rand(1, 3, 17, 25), sparse)
        for depth in received + decoded:
            anchors = sparse_depth_at_resolution(sparse, depth.shape[-2:])
            accepted = anchors > 0
            self.assertTrue(torch.equal(depth[accepted], anchors[accepted]))
        self.assertEqual(model.n_propagation_iterations, (16, 8, 4, 2, 2))
        self.assertEqual(
            [block.n_propagation_iterations for block in model.decoder],
            [8, 4, 2, 2])


if __name__ == '__main__':
    unittest.main()
