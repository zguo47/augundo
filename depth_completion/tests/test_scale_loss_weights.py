import os
import runpy
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from depth_completion_debug_analysis import StageDiagnostics


class ScaleLossWeightsTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = ScalePropagationDepthCompletionModel.__new__(ScalePropagationDepthCompletionModel)
        self.wrapper.min_predict_depth = 0.1
        self.wrapper.max_predict_depth = 100.
        self.target = torch.tensor([[[[1., 2., 0.], [4., 6., 8.]]]])

    def outputs(self):
        geometry = [torch.full_like(self.target, 1. + r / 10., requires_grad=True)
                    for r in range(5)]
        metric = [torch.full_like(self.target, 3. + r / 10., requires_grad=True)
                  for r in range(5)]
        return [metric[0]] + geometry + [torch.ones_like(self.target) for _ in range(5)] + metric

    def test_defaults_preserve_existing_loss(self):
        outputs = self.outputs()
        loss, info = self.wrapper.compute_loss_supervised(self.target, outputs, {})
        explicit, _ = self.wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_supervised=1., w_si=1., w_log_l1=1.))
        torch.testing.assert_close(loss, info['loss_si'] + info['loss_log_l1'])
        torch.testing.assert_close(loss, explicit)

    def test_independent_weights_and_overall_supervision_weight(self):
        outputs = self.outputs()
        _, baseline = self.wrapper.compute_loss_supervised(self.target, outputs, {})
        loss, info = self.wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_supervised=2., w_si=3., w_log_l1=0.25))
        torch.testing.assert_close(loss, 6. * baseline['loss_si'] + 0.5 * baseline['loss_log_l1'])
        torch.testing.assert_close(info['loss_si'], baseline['loss_si'])
        torch.testing.assert_close(info['loss_log_l1'], baseline['loss_log_l1'])
        torch.testing.assert_close(info['loss_si_weighted'], 6. * info['loss_si'])
        torch.testing.assert_close(info['loss_log_l1_weighted'], 0.5 * info['loss_log_l1'])
        for name in ['si', 'log_l1']:
            expected = sum(info['loss_{}_r{}'.format(name, r)] / (4 ** r) for r in range(5))
            torch.testing.assert_close(info['loss_' + name], expected)

    def test_each_objective_can_be_disabled_independently(self):
        for w_si, w_log in [(0., 1.), (1., 0.), (0., 0.)]:
            outputs = self.outputs()
            loss, info = self.wrapper.compute_loss_supervised(
                self.target, outputs, dict(w_si=w_si, w_log_l1=w_log))
            torch.testing.assert_close(loss, w_si * info['loss_si'] + w_log * info['loss_log_l1'])
            loss.backward()
            for maps, weight in [(outputs[1:6], w_si), (outputs[11:16], w_log)]:
                for value in maps:
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertEqual(float(value.grad.abs().sum()) > 0., weight > 0.)

    def test_gradients_follow_independent_weights(self):
        outputs = self.outputs()
        baseline, _ = self.wrapper.compute_loss_supervised(self.target, outputs, {})
        tensors = outputs[1:6] + outputs[11:16]
        expected = torch.autograd.grad(baseline, tensors, retain_graph=True)
        weighted, _ = self.wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_si=3., w_log_l1=0.5))
        actual = torch.autograd.grad(weighted, tensors)
        for index, (a, b) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(a, b * (3. if index < 5 else 0.5))

    def test_debug_probe_compares_weighted_gradient_contributions(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        si = model.weight.square().sum()
        metric = 3. * model.weight.square().sum()
        info = dict(loss_si=si, loss_log_l1=metric,
                    loss_si_weighted=3. * si, loss_log_l1_weighted=0.5 * metric)
        debug = StageDiagnostics.__new__(StageDiagnostics)
        debug.model = model
        debug.context = dict(step=1)
        debug.probe_interval = 1000
        debug.probing = False
        records = []
        debug.emit = lambda event, **fields: records.append(fields)
        debug.gradient_objectives(info)
        group = records[0]['groups']['weight']
        self.assertAlmostEqual(group['si_norm'], 6.)
        self.assertAlmostEqual(group['metric_norm'], 3.)
        self.assertIsNone(model.weight.grad)
        (info['loss_si_weighted'] + info['loss_log_l1_weighted']).backward()
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 9.))

    def test_training_cli_accepts_loss_weight_dictionary(self):
        script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_depth_completion.py')
        argv = [script, '--train_images_path', 'images.txt',
                '--train_sparse_depth_path', 'sparse.txt', '--device', 'cpu',
                '--checkpoint_path', 'outputs',
                '--w_losses', 'w_supervised=2.0', 'w_si=3.0', 'w_log_l1=0.5']
        with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.train') as train:
            runpy.run_path(script, run_name='__main__')
        self.assertEqual(train.call_args.kwargs['w_losses'],
                         dict(w_supervised=2., w_si=3., w_log_l1=0.5))


if __name__ == '__main__':
    unittest.main()
