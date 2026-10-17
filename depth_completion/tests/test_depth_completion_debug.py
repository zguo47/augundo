import copy
import json
import os
import sys
import tempfile
import unittest
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from depth_completion_debug import DepthCompletionDebug
from scale_propagation_model import ScalePropagationDepthModel


class DepthDebugTest(unittest.TestCase):
    def make_debug(self, model, directory):
        wrapper = SimpleNamespace(
            min_predict_depth=1.5, max_predict_depth=100.,
            model=SimpleNamespace(model_depth=model))
        return DepthCompletionDebug(wrapper, directory, interval=1, max_samples=1)

    def test_observation_preserves_training_and_rng(self):
        torch.manual_seed(1)
        for rejection in (True, False):
            reference = ScalePropagationDepthModel(
                n_channels=8, min_predict_depth=1.5, max_predict_depth=100.,
                use_outlier_rejection=rejection)
            observed = copy.deepcopy(reference)
            image = torch.rand(1, 3, 17, 25)
            sparse = torch.zeros(1, 1, 17, 25)
            sparse[:, :, ::3, ::3] = 10.
            target = torch.ones_like(sparse) * 12.
            initial_rng = torch.get_rng_state()
            expected = reference(image, sparse)
            expected.mean().backward()
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(initial_rng)
            with tempfile.TemporaryDirectory() as directory:
                debug = self.make_debug(observed, directory)
                debug.begin('train', 1)
                debug.inputs('network_input', image, sparse, target)
                debug.sparse_levels(sparse)
                actual = observed(image, sparse)
                debug.training_prediction(actual, target, sparse, {'loss': actual.mean()})
                actual.mean().backward()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for a, b in zip(reference.parameters(), observed.parameters()):
                    if a.grad is None:
                        self.assertIsNone(b.grad)
                    else:
                        self.assertTrue(torch.equal(a.grad, b.grad))
                optimizer = torch.optim.Adam(observed.parameters(), lr=1e-4)
                debug.before_step(optimizer)
                optimizer.step()
                debug.after_step()
                debug.close()
                with open(debug.path) as stream:
                    records = [json.loads(line) for line in stream]
                self.assertEqual(sum(r['event'] == 'decoder' for r in records), 4)
                self.assertTrue(any(r['event'] == 'output_gradient' for r in records))
                gradients = next(r for r in records if r['event'] == 'gradients_before_optimizer')
                self.assertEqual(set(gradients['rejection_heads']),
                                 {'bottom', 'decoder.0', 'decoder.1', 'decoder.2', 'decoder.3'})
                if rejection:
                    self.assertEqual(sum(r['event'] == 'rejection' for r in records), 5)
                    self.assertTrue(any(g['grad_norm'] > 0.
                                        for g in gradients['rejection_heads'].values()))
                else:
                    self.assertFalse(any(r['event'] == 'rejection' for r in records))

    def test_inverse_failure_and_worst_sample_capture(self):
        model = ScalePropagationDepthModel(n_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            image = torch.zeros(1, 3, 16, 16)
            sparse = torch.zeros(1, 1, 16, 16)
            target = torch.ones_like(sparse) * 10.
            prediction = target.clone()
            prediction[0, 0, 2, 3] = 0.
            dataset = SimpleNamespace(image_paths=['example.png'])
            debug.start_validation(1000)
            debug.validation_sample(0, image, sparse, target, prediction, 0., 100., dataset)
            debug.finish_validation()
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            failure = next(r for r in records if r['event'] == 'inverse_failure')
            self.assertEqual(failure['coordinates_yx'], [[2, 3]])
            self.assertEqual(failure['prediction_values'], [0.])
            self.assertEqual(failure['native_imae'], 'inf')
            saved = np.load(os.path.join(directory, 'debug', 'inverse_failure_0.npz'))
            self.assertEqual(saved['prediction'][2, 3], 0.)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'debug', 'worst_0.npz')))

    def test_sparse_counts_and_rejection_ratios_use_actual_selection(self):
        model = ScalePropagationDepthModel(
            n_channels=8, use_outlier_rejection=False).eval()
        effective_weights = []
        handles = [stage.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: effective_weights.append(
                torch.sigmoid(inputs[4]) * inputs[5]))
            for stage in [model] + list(model.decoder)]
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            sparse = torch.zeros(1, 1, 16, 16)
            # Nearest sampling misses this point at every coarse level.
            sparse[0, 0, 7, 7] = 5.
            debug.begin('validation', 1)
            debug.sparse_levels(sparse)
            with torch.no_grad():
                model(torch.rand(1, 3, 16, 16), sparse)
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            levels = [r for r in records if r['event'] == 'sparse_sampling']
            self.assertEqual([r['measurement_count'] for r in levels], [1] * 5)
            blocks = [r for r in records if r['event'] == 'decoder']
            self.assertEqual([r['keep_fraction_at_sparse'] for r in blocks], [1.] * 4)
            self.assertEqual([r['propagation_iterations'] for r in blocks], [2, 3, 4, 6])
            scales = [r for r in records if r['event'] == 'scale_propagation']
            self.assertEqual(len(scales), 5)
            self.assertTrue(all(not r['restores_anchors'] for r in scales))
            self.assertTrue(all(r['propagation_activation'] == 'sigmoid' for r in scales))
            for record, weights in zip(scales, effective_weights):
                self.assertIsNotNone(record['surface_compatibility'])
                self.assertAlmostEqual(record['outgoing_weight_sum']['mean'],
                                       float(weights.sum(dim=1).mean()), places=6)
            for record, weights in zip(blocks, effective_weights[1:]):
                self.assertAlmostEqual(record['outgoing_weight_mean'],
                                       float(weights.mean()), places=6)
            self.assertTrue(all('scaleless_depth' in r and 'scale_map' in r for r in blocks))
        for handle in handles:
            handle.remove()

    def test_empty_ground_truth_does_not_crash_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(ScalePropagationDepthModel(n_channels=8), directory)
            image = torch.zeros(1, 3, 16, 16)
            depth = torch.zeros(1, 1, 16, 16)
            debug.start_validation(1)
            debug.validation_sample(0, image, depth, depth, depth, 0., 100., SimpleNamespace())
            debug.finish_validation()
            debug.close()

    def test_rejection_quality_uses_selected_original_measurement_coordinates(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        with torch.no_grad():
            for stage in [model] + list(model.decoder):
                stage.rejection_map.conv.weight.zero_()
                stage.rejection_bias.copy_(torch.tensor([100., -100.]).reshape(1, 2, 1, 1))
            model.decoder[0].rejection_bias.copy_(
                torch.tensor([-100., 100.]).reshape(1, 2, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            sparse = torch.zeros(1, 1, 16, 16)
            target = torch.zeros_like(sparse)
            sparse[0, 0, 7, 7] = 20.
            sparse[0, 0, 7, 15] = 5.
            target[0, 0, 7, 7] = 5.
            target[0, 0, 7, 15] = 5.
            image = torch.rand(1, 3, 16, 16)
            debug.begin('validation', 1)
            with torch.no_grad():
                model(image, sparse)
            debug.inputs('validation_input', image, sparse, target)
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            quality = {r['level']: r for r in records if r['event'] == 'rejection_quality'}
            self.assertEqual(len(quality), 5)
            bottom = quality['bottom']['categories']
            self.assertEqual(bottom['all_with_ground_truth']['count'], 1)
            self.assertEqual(bottom['disagreement_over_3m']['count'], 1)
            self.assertEqual(bottom['disagreement_over_3m']['rejection_fraction'], 0.)
            coarse = quality['decoder.0']['categories']
            self.assertEqual(coarse['accurate_within_0_1m']['count'], 1)
            self.assertEqual(coarse['accurate_within_0_1m']['rejection_fraction'], 1.)
            self.assertEqual(coarse['disagreement_over_3m']['rejection_fraction'], 1.)
            finest = quality['decoder.3']['categories']
            self.assertEqual(finest['all_with_ground_truth']['count'], 2)
            self.assertEqual(finest['disagreement_over_3m']['rejection_fraction'], 0.)
            margins = [r for r in records if r['event'] == 'rejection']
            self.assertEqual(margins[0]['keep_minus_reject_at_sparse']['mean'], 200.)
            self.assertEqual(margins[1]['keep_minus_reject_at_sparse']['mean'], -200.)
            self.assertAlmostEqual(margins[0]['scaled_sparse_input']['max'], 0.2)

    def test_cli_accumulation_validation_and_disabled_equivalence(self):
        import matplotlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ['images', 'sparse', 'target']:
                folder = root / name
                folder.mkdir()
                path = folder / '0000000000.png'
                if name == 'images':
                    values = np.full((32, 48, 3), 128, dtype=np.uint8)
                else:
                    values = np.full((32, 48), 3072, dtype=np.uint16)
                    if name == 'sparse':
                        values[:] = 0
                        values[::4, ::4] = 2560
                Image.fromarray(values).save(path)
                paths[name] = path
            paths['intrinsics'] = root / 'intrinsics.npy'
            np.save(paths['intrinsics'], np.eye(3, dtype=np.float32))
            for name, path in paths.items():
                (root / (name + '.txt')).write_text((str(path) + '\n') * 2)
            arguments = [
                '--model_name', 'scale_propagation_kitti', '--supervision_type', 'supervised',
                '--w_losses', 'w_supervised=1.0',
                '--use_outlier_rejection', 'false', '--min_predict_depth', '1.5',
                '--max_predict_depth', '100', '--n_batch', '1', '--n_height', '32', '--n_width', '48',
                '--learning_rates', '0.0001', '--learning_schedule', '1',
                '--n_step_per_gradient_accumulation', '2', '--n_step_per_summary', '1',
                '--n_step_per_checkpoint', '1', '--start_step_validation', '1',
                '--min_evaluate_depth', '0', '--max_evaluate_depth', '100',
                '--debug_interval', '1', '--debug_max_samples', '1', '--device', 'cpu', '--n_thread', '0']
            for flag, name in [
                    ('train_images_path', 'images'), ('train_sparse_depth_path', 'sparse'),
                    ('train_ground_truth_path', 'target'), ('train_intrinsics_path', 'intrinsics'),
                    ('val_image_path', 'images'), ('val_sparse_depth_path', 'sparse'),
                    ('val_ground_truth_path', 'target'), ('val_intrinsics_path', 'intrinsics')]:
                arguments.extend(['--' + flag, str(root / (name + '.txt'))])
            script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_depth_completion.py')
            sys.path.insert(0, os.path.dirname(script))
            for enabled in ['false', 'true']:
                torch.manual_seed(3)
                np.random.seed(3)
                argv = [script] + arguments + ['--debug_depth', enabled,
                                               '--checkpoint_path', str(root / enabled)]
                # The repository targets older Pillow; adapt only this test's environment.
                lanczos = getattr(Image, 'Resampling', Image).LANCZOS
                get_cmap = getattr(
                    matplotlib.cm, 'get_cmap', lambda name: matplotlib.colormaps[name])
                with mock.patch.object(Image, 'ANTIALIAS', lanczos, create=True), \
                        mock.patch.object(matplotlib.cm, 'get_cmap', get_cmap, create=True), \
                        mock.patch.object(sys, 'argv', argv):
                    runpy.run_path(script, run_name='__main__')
            records = [json.loads(line) for line in
                       (root / 'true/debug/diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['event'] == 'gradients_before_optimizer' for r in records), 1)
            self.assertEqual(sum(r['event'] == 'validation_sample' for r in records), 2)
            self.assertFalse((root / 'false/debug').exists())
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            for enabled in ['false', 'true']:
                for phase in ['train', 'val']:
                    events = EventAccumulator(str(root / enabled / ('events-' + phase))).Reload()
                    prefix = 'train' if phase == 'train' else 'eval'
                    self.assertIn(prefix + '/scaleless_depths/r0', events.Tags()['images'])
                    self.assertIn(prefix + '/scale_maps/r4', events.Tags()['images'])
                    self.assertIn(prefix + '/scaleless_comparison/r0', events.Tags()['images'])
                    self.assertIn(prefix + '/scaleless_comparison/r4', events.Tags()['images'])
                    if phase == 'train':
                        self.assertIn('train_loss_si_r3', events.Tags()['scalars'])
                        self.assertIn('train_loss_log_l1_r0', events.Tags()['scalars'])
                        self.assertIn('train_loss_si_r4', events.Tags()['scalars'])
                        self.assertIn('train_loss_log_l1_r4', events.Tags()['scalars'])
            checkpoints = []
            for enabled in ['false', 'true']:
                path = next((root / enabled).glob('checkpoints-1*'))
                if path.is_dir():
                    path = next(path.glob('scale-propagation-*.pth'))
                checkpoints.append(torch.load(str(path), map_location='cpu')['model_state_dict'])
            for name in checkpoints[0]:
                self.assertTrue(torch.equal(checkpoints[0][name], checkpoints[1][name]), name)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
