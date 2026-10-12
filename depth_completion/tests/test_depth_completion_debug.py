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
                debug.pooling(sparse)
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

    def test_empty_ground_truth_does_not_crash_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(ScalePropagationDepthModel(n_channels=8), directory)
            image = torch.zeros(1, 3, 16, 16)
            depth = torch.zeros(1, 1, 16, 16)
            debug.start_validation(1)
            debug.validation_sample(0, image, depth, depth, depth, 0., 100., SimpleNamespace())
            debug.finish_validation()
            debug.close()

    def test_cli_accumulation_validation_and_disabled_equivalence(self):
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
                with mock.patch.object(Image, 'ANTIALIAS', lanczos, create=True), \
                        mock.patch.object(sys, 'argv', argv):
                    runpy.run_path(script, run_name='__main__')
            records = [json.loads(line) for line in
                       (root / 'true/debug/diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['event'] == 'gradients_before_optimizer' for r in records), 1)
            self.assertEqual(sum(r['event'] == 'validation_sample' for r in records), 2)
            self.assertFalse((root / 'false/debug').exists())
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
