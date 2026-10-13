import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scale_propagation_model import (
    ScalePropagationDepthModel,
    propagate_scale,
    rejection_keep_mask,
    sparse_depth_at_resolution)


class HierarchicalScalePropagationTest(unittest.TestCase):

    def test_nearest_copies_values_instead_of_averaging(self):
        sparse = torch.tensor([[[[2., 40., 0., 60.],
                                [70., 80., 90., 100.],
                                [0., 110., 6., 120.],
                                [130., 140., 150., 160.]]]])
        sampled = sparse_depth_at_resolution(sparse, (2, 2))
        self.assertTrue(torch.equal(
            sampled, torch.tensor([[[[2., 0.], [0., 6.]]]])))
        self.assertTrue(torch.equal(
            sparse_depth_at_resolution(sparse, (4, 4)), sparse))

    def test_one_source_sends_to_the_right(self):
        depth = torch.full((1, 1, 5, 5), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float()
        logits = torch.full((1, 9, 5, 5), -100.)
        logits[0, 5, 2, 2] = 100.
        result = propagate_scale(depth, sparse, keep, logits)
        expected = depth.clone()
        # Source scale is 6 / 2 = 3. The destination combines that message
        # with its unit prior: 2 * (1 + 3) / (1 + 1) = 4.
        expected[0, 0, 2, 3] = 4.
        torch.testing.assert_close(result, expected)

        # Weights at non-measurement pixels cannot send messages.
        logits[0, :, 1, 1] = 100.
        torch.testing.assert_close(
            propagate_scale(depth, sparse, keep, logits), expected)

    def test_outgoing_boundary_messages_do_not_wrap(self):
        depth = torch.full((1, 1, 3, 3), 2.)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 0, 0] = 6.
        logits = torch.full((1, 9, 3, 3), -100.)
        logits[0, 0, 0, 0] = 100.
        torch.testing.assert_close(
            propagate_scale(depth, sparse, (sparse > 0).float(), logits),
            depth)

    def test_rejected_and_empty_sources_preserve_dense_prediction(self):
        depth = torch.rand(2, 1, 5, 7) + 1.
        sparse = torch.full_like(depth, 50.)
        logits = torch.zeros(2, 9, 5, 7)
        keep = torch.zeros_like(depth, requires_grad=True)
        result = propagate_scale(depth, sparse, keep, logits)
        self.assertTrue(torch.equal(result, depth))
        result.sum().backward()
        self.assertTrue(torch.isfinite(keep.grad).all())
        self.assertGreater(float(keep.grad.abs().sum()), 0.)
        sparse.zero_()
        self.assertTrue(torch.equal(
            propagate_scale(depth, sparse, torch.zeros_like(depth), logits),
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
            # which nearest sampling can miss at lower resolutions.
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


if __name__ == '__main__':
    unittest.main()
