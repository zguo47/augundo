import os
import sys
import unittest

import torch


REPOSITORY_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'))
DEPTH_COMPLETION_SOURCE = os.path.join(
    REPOSITORY_ROOT,
    'depth_completion',
    'src')

if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)
if DEPTH_COMPLETION_SOURCE not in sys.path:
    sys.path.insert(0, DEPTH_COMPLETION_SOURCE)

from partition_attention_model_patch_token_ablation import \
    PartitionAttentionPatchTokenDepthModel


class PartitionAttentionPatchTokenAblationTest(unittest.TestCase):

    def test_forward_backward(self):
        model = PartitionAttentionPatchTokenDepthModel(
            min_predict_depth=0.1,
            max_predict_depth=8.0,
            n_channels=16,
            n_head=4,
            patch_size=4,
            partition_size=16,
            n_partition=4)
        image = torch.rand(1, 3, 64, 64)

        output = model(image)

        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), 0.1)
        self.assertLessEqual(output.max().item(), 8.0)

        output.mean().backward()
        self.assertIsNotNone(model.patch_projection.weight.grad)
        self.assertIsNotNone(
            model.pixel_from_patch_attention.query_projection.weight.grad)
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
