import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.src.geometry_losses import scale_aligned_l1, affine_invariant_loss, fit_affine_points
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from utils.src import loss_utils


class GeometryLossTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_scale_fit_is_invariant_and_masked(self):
        z = torch.tensor([[[[1., 2., 3., 90.]]]], requires_grad=True)
        gt = torch.tensor([[[[4., 8., 13., 0.]]]])
        mask = gt > 0
        loss, a = scale_aligned_l1(z, gt, mask)
        other, _ = scale_aligned_l1(z * 7, gt, mask)
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[0, 0, 0, 3]), 0.)
        self.assertGreater(float(z.grad.abs().sum()), 0.)

    def test_affine_solver_retains_z_offset(self):
        src = torch.randn(12, 3)
        src[:, 2] = src[:, 2].abs() + 2
        src.requires_grad_()
        gt = src.detach() * 2 + torch.tensor([0., 0., 3.])
        a, b = fit_affine_points(src, gt, torch.ones(12))
        torch.testing.assert_close(a, torch.tensor(2.))
        torch.testing.assert_close(b, torch.tensor(3.))
        (a + b).backward()
        self.assertTrue(torch.isfinite(src.grad).all())

    def test_affine_xyz_loss_and_empty_mask(self):
        z = (torch.rand(1, 1, 6, 8) + 1).requires_grad_()
        gt = z.detach() * 3
        k = torch.tensor([[[8., 0., 3.5], [0., 8., 2.5], [0., 0., 1.]]])
        valid = torch.ones_like(z, dtype=torch.bool)
        loss, a, b = affine_invariant_loss(z, gt, valid, k, resolution=4)
        self.assertLess(float(loss), 1e-6)
        torch.testing.assert_close(a, torch.tensor([3.]))
        gt[0, 0, 2, 3] += 1
        loss, _, _ = affine_invariant_loss(z, gt, valid, k, resolution=4)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        empty, _, _ = affine_invariant_loss(z, gt, valid & False, k, resolution=4)
        self.assertEqual(float(empty), 0.)

    def test_wrapper_weights_and_geometry_gradients(self):
        wrapper = ScalePropagationDepthCompletionModel.__new__(ScalePropagationDepthCompletionModel)
        wrapper.min_predict_depth, wrapper.max_predict_depth = .1, 8.
        geometries = [(torch.rand(1, 1, 6, 8) + 1).requires_grad_() for _ in range(5)]
        metrics = [(torch.rand(1, 1, 6, 8) + 2).requires_grad_() for _ in range(5)]
        gt = torch.rand(1, 1, 6, 8) * 3 + 1
        gt[..., 0, 0] = 0
        k = torch.tensor([[[8., 0., 3.5], [0., 8., 2.5], [0., 0., 1.]]])
        outputs = [metrics[0]] + geometries + [torch.ones_like(gt)] * 5 + metrics
        # A small fit grid keeps this unit test fast; the public default is 64.
        def small_fit(z, g, m, camera):
            return affine_invariant_loss(z, g, m, camera, resolution=4)
        with patch.object(loss_utils, 'affine_invariant_loss', side_effect=small_fit):
            loss, info = wrapper.compute_loss_supervised(gt, outputs,
                dict(w_supervised=2., w_si=0., w_log_l1=0., w_affine=.3, w_scale_aligned_l1=.7), k)
        expected = sum((.3 * info['loss_affine_r{}'.format(r)] +
                        .7 * info['loss_scale_aligned_l1_r{}'.format(r)]) / 4**r for r in range(5)) * 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for z in geometries:
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertTrue(all(float(metric.grad.abs().sum()) == 0. for metric in metrics))
        _, default = wrapper.compute_loss_supervised(gt, outputs, {})
        torch.testing.assert_close(default['loss'], default['loss_si'] + default['loss_log_l1'])
        self.assertEqual(float(default['loss_affine']), 0.)
        self.assertEqual(float(default['loss_scale_aligned_l1']), 0.)


if __name__ == '__main__':
    unittest.main()
