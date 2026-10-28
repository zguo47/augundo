import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import depth_completion as pipeline
from fixed_geometry_scale_propagation_model import FixedGeometryScalePropagationDepthModel
from fixed_geometry_scale_propagation_depth_completion_model import FixedGeometryScalePropagationDepthCompletionModel
from scale_propagation_model import ScalePropagationDepthModel
from depth_completion_model import DepthCompletionModel
from depth_completion_debug import DepthCompletionDebug


class ScaleAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.image = torch.rand(2, 3, 17, 25)
        self.sparse = torch.zeros(2, 1, 17, 25)
        self.sparse[:, :, 4, 7] = 2.0
        self.sparse[:, :, 12, 20] = 6.0
        self.target = 1 + torch.rand_like(self.sparse) * 6

    def wrapper(self):
        return FixedGeometryScalePropagationDepthCompletionModel(
            device=torch.device('cpu'), min_predict_depth=0.1,
            max_predict_depth=100, n_propagation_iterations=(1, 1, 1, 1, 1),
            n_scale_refinement=1, use_outlier_rejection=False)

    def test_zero_initialized_adaptation_matches_original_and_runs_before_propagation(self):
        torch.manual_seed(11)
        old = ScalePropagationDepthModel(
            n_channels=8, n_propagation_iterations=(1,)*5,
            n_scale_refinement=1, use_outlier_rejection=False)
        torch.manual_seed(11)
        model = FixedGeometryScalePropagationDepthModel(
            n_channels=8, n_propagation_iterations=(1,)*5,
            n_scale_refinement=1, use_outlier_rejection=False)
        events = []
        block = model.decoder[-1]
        block.scale_update.register_forward_hook(lambda *args: events.append('adapt'))
        block.scale_propagation.register_forward_hook(lambda *args: events.append('propagate'))
        out = model(self.image, self.sparse, return_all_outputs=True)
        reference = old(self.image, self.sparse, return_all_outputs=True)
        torch.testing.assert_close(out['depth'], reference['depth'])
        self.assertEqual(events, ['adapt', 'propagate'])
        self.assertEqual([z.shape[-2:] for z in out['scaleless_depths']],
                         [(17,25), (9,13), (5,7), (3,4), (2,2)])
        for z in out['scaleless_depths']:
            torch.testing.assert_close(z.mean((2,3)), torch.ones(2,1))

    def test_adapted_scale_reaches_propagation_with_detached_geometry_inputs(self):
        wrapper = self.wrapper()
        block = wrapper.model_depth.decoder[-1]
        with torch.no_grad():
            block.scale_update.convolutions[-1].conv.weight.fill_(0.01)
        captured = {}
        def adaptation(module, inputs, output):
            captured['input'] = inputs[0]
            captured['expected'] = inputs[0][:,-4:-3] * torch.exp(torch.tanh(output))
        block.scale_update.register_forward_hook(adaptation)
        block.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: captured.update(actual=inputs[0]))
        wrapper.forward_depth(self.image, self.sparse)
        torch.testing.assert_close(captured['expected'], captured['actual'])
        self.assertFalse(torch.equal(captured['actual'], captured['input'][:,-4:-3]))

    def test_losses_keep_geometry_detached_and_train_both_heads(self):
        wrapper = self.wrapper()
        outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
        self.assertEqual(len(outputs), 16)
        loss, info = wrapper.compute_loss_supervised(self.target, outputs, {})
        params = [wrapper.model_depth.coarse_scaleless_depth.conv.weight] + [
            block.depth_update.conv.weight for block in wrapper.model_depth.decoder]
        grads = torch.autograd.grad(info['loss_log_l1'], params, allow_unused=True,
                                    retain_graph=True)
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grads))
        loss.backward()
        self.assertGreater(float(params[-1].grad.abs().sum()), 0)
        scale_head = wrapper.model_depth.decoder[-1].scale_update.convolutions[-1].conv.weight
        self.assertGreater(float(scale_head.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_repository_flags_checkpoint_and_debug_stages(self):
        wrapper = DepthCompletionModel(
            model_name='fixed_geometry_scale_propagation_kitti',
            network_modules=[], min_predict_depth=0.1, max_predict_depth=100,
            device=torch.device('cpu'), use_outlier_rejection=False,
            n_scale_refinement=1, n_propagation_iterations=(1,)*5)
        with tempfile.TemporaryDirectory() as directory:
            debug = DepthCompletionDebug(wrapper, directory, interval=1, max_samples=1)
            debug.begin('train', 1)
            outputs = wrapper.forward_depth(self.image, self.sparse, None,
                                            return_all_outputs=True)
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(len([r for r in records if r['event']=='scale_adaptation']), 4)
            self.assertEqual(len([r for r in records if r['event']=='scale_refinement']), 1)
            debug.begin('validation', 1, sample=0)
            wrapper.forward_depth(self.image[:1], self.sparse[:1], None)
            for level in ['decoder.0', 'decoder.1', 'decoder.2', 'decoder.3']:
                stages = [record['stage'] for record in debug.trace[level]]
                self.assertEqual(stages, [
                    'before_upsampling', 'after_upsampling',
                    'after_geometry_update', 'scale_adaptation_1',
                    'after_scale_adaptation', 'after_propagation'])
            debug.close()
            path = os.path.join(directory, 'model.pth')
            wrapper.model.save_model(path, step=7)
            wrapper.model.restore_model(path)
            self.assertEqual(len(outputs), 16)

    def test_rejection_and_optional_sparse_inputs_still_work(self):
        model = FixedGeometryScalePropagationDepthModel(
            n_channels=8, use_outlier_rejection=True, use_sparse_depth_features=True,
            n_propagation_iterations=(1,)*5, n_scale_refinement=1)
        output = model(self.image, self.sparse, return_rejection_outputs=True)
        self.assertEqual(len(output['rejection_logits']), 5)
        self.assertTrue(torch.isfinite(output['depth']).all())


if __name__ == '__main__':
    unittest.main()
