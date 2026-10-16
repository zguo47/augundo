import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from depth_completion_model import DepthCompletionModel


class ScalelessSummaryTest(unittest.TestCase):

    def setUp(self):
        self.model = DepthCompletionModel.__new__(DepthCompletionModel)
        self.model.max_predict_depth = 10.

    def display(self, geometry, target):
        image = torch.rand(geometry.shape[0], 3, *geometry.shape[-2:])
        writer = mock.Mock()
        with mock.patch('depth_completion_model.log_utils.colorize',
                        side_effect=lambda value, **kwargs: value.repeat(1, 3, 1, 1)), \
                mock.patch('depth_completion_model.torchvision.utils.make_grid',
                           side_effect=lambda value, **kwargs: value) as grid:
            self.model.log_scaleless_comparison(
                writer, 'eval', 17, 0, image, geometry, target)
        self.assertEqual(writer.add_image.call_args.args[0], 'eval/scaleless_comparison/r0')
        self.assertEqual(grid.call_args.kwargs['nrow'], 1)
        self.assertEqual(writer.add_image.call_args.kwargs['global_step'], 17)
        return image, writer.add_image.call_args.args[1].chunk(5, dim=-1)

    def test_aligns_each_image_and_preserves_five_column_order(self):
        geometry = torch.tensor([[[[0.5, 1., 1.5], [1., 0.5, 1.5]]]]).repeat(2, 1, 1, 1)
        geometry.requires_grad_()
        target = geometry.detach() * torch.tensor([3., 5.]).reshape(2, 1, 1, 1)
        target[:, :, 0, 0] = 0.
        original = geometry.detach().clone()
        image, panels = self.display(geometry, target)
        rgb, raw, truth, aligned, error = panels
        valid = (target > 0).expand_as(truth)
        torch.testing.assert_close(rgb, image)
        torch.testing.assert_close(raw, (geometry.detach() / 2.).repeat(1, 3, 1, 1))
        torch.testing.assert_close(truth, (target / 10.).repeat(1, 3, 1, 1))
        torch.testing.assert_close(aligned[valid], truth[valid])
        self.assertLess(float(error.max()), 1e-5)
        self.assertFalse(aligned.requires_grad)
        self.assertIsNone(geometry.grad)
        self.assertTrue(torch.equal(geometry, original))

    def test_relative_error_uses_aligned_depth_and_fixed_twenty_percent_range(self):
        geometry = torch.ones(1, 1, 2, 2)
        target = torch.tensor([[[[4., 4.84], [0., 0.]]]])
        _, panels = self.display(geometry, target)
        aligned, error = panels[3], panels[4]
        self.assertAlmostEqual(float(aligned[0, 0, 0, 0]), 0.44, places=6)
        self.assertAlmostEqual(float(error[0, 0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(error[0, 0, 0, 1]), (4.84 - 4.4) / 4.84 / .2, places=6)
        self.assertTrue((error[:, :, 1] == 0).all())

    def test_empty_ground_truth_has_black_truth_and_error_without_nan(self):
        _, panels = self.display(torch.ones(1, 1, 2, 3), torch.zeros(1, 1, 2, 3))
        self.assertTrue(all(torch.isfinite(value).all() for value in panels))
        self.assertTrue((panels[2] == 0).all())
        self.assertTrue((panels[4] == 0).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
