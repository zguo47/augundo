'''Optional diagnostics; observations never replace training tensors or metrics.'''

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def statistics(value):
    value = array(value)
    finite = np.isfinite(value)
    values = value[finite].astype(np.float64)
    result = dict(shape=list(value.shape), dtype=str(value.dtype),
                  count=int(value.size), nonfinite=int((~finite).sum()),
                  zero=int((value == 0).sum()), negative=int((value < 0).sum()))
    if values.size:
        sample = values[::max(1, values.size // 65536)]
        result.update(min=float(values.min()), max=float(values.max()),
                      mean=float(values.mean()), std=float(values.std()),
                      quantiles_approx_p01_p50_p99=np.percentile(
                          sample, [1, 50, 99]).tolist())
    return result


def errors(prediction, target, mask):
    prediction = array(prediction)[mask].astype(np.float64)
    target = array(target)[mask].astype(np.float64)
    finite = np.isfinite(prediction) & np.isfinite(target)
    result = dict(count=int(target.size), nonfinite=int((~finite).sum()),
                  nonpositive_prediction=int((prediction <= 0).sum()))
    if not finite.any():
        return result
    prediction, target = prediction[finite], target[finite]
    delta = prediction - target
    absolute = np.abs(delta)
    squared = delta ** 2
    tail_count = max(1, int(np.ceil(squared.size * 0.01)))
    tail = np.partition(squared, squared.size - tail_count)[-tail_count:]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inverse = 1000.0 / prediction - 1000.0 / target
        result.update(imae_1_per_km=float(np.mean(np.abs(inverse))),
                      irmse_1_per_km=float(np.sqrt(np.mean(inverse ** 2))))
    result.update(
        finite_count=int(delta.size), abs_sum_m=float(absolute.sum()),
        squared_sum_m2=float(squared.sum()), mae_mm=float(1000 * absolute.mean()),
        rmse_mm=float(1000 * np.sqrt(squared.mean())), bias_m=float(delta.mean()),
        error_p50_p90_p99_m=np.percentile(absolute, [50, 90, 99]).tolist(),
        fraction_over_1m=float(np.mean(absolute > 1)),
        fraction_over_3m=float(np.mean(absolute > 3)),
        top_1pct_squared_error_share=float(tail.sum() / (squared.sum() + 1e-30)),
        mean_log1p_abs_error=float(np.log1p(absolute).mean()),
        mean_log1p_gradient_factor=float((1 / (1 + absolute)).mean()))
    positive = target > 0
    if positive.any():
        result['median_prediction_target_ratio'] = float(np.median(
            prediction[positive] / target[positive]))
    return result


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class DepthCompletionDebug(object):
    '''Write sampled internals and exhaustive validation error diagnostics.'''

    def __init__(self, wrapper, checkpoint_path, interval=500, max_samples=3):
        self.directory = os.path.join(checkpoint_path, 'debug')
        os.makedirs(self.directory, exist_ok=True)
        self.path = os.path.join(self.directory, 'diagnostics.jsonl')
        self.interval = max(1, interval)
        self.max_samples = max(0, max_samples)
        self.minimum = wrapper.min_predict_depth
        self.maximum = wrapper.max_predict_depth
        self.model = wrapper.model.model_depth
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.active = False
        self.lock = threading.RLock()
        self.context = {}
        self.handles = []
        self.before = {}
        self.training_shape = None
        self.worst = []
        self.buckets = {}
        self.inverse_failures = 0
        self.register_hooks()

    def emit(self, event, **values):
        with self.lock:
            record = dict(time=time.time(), event=event, **self.context)
            record.update(values)
            with open(self.path, 'a') as stream:
                stream.write(json.dumps(serializable(record), allow_nan=False) + '\n')

    def provenance(self, settings, datasets):
        self.training_shape = settings.get('train_shape')
        sources = {}
        root = os.path.dirname(__file__)
        for path in [inspect.getfile(type(self.model)), __file__,
                     os.path.join(root, 'depth_completion.py'),
                     os.path.join(root, 'scale_propagation_depth_completion_model.py'),
                     os.path.join(root, 'datasets.py'),
                     os.path.join(root, 'train_depth_completion.py'),
                     os.path.abspath('utils/src/eval_utils.py'),
                     os.path.abspath('utils/src/loss_utils.py'),
                     os.path.abspath('utils/src/data_utils.py')]:
            with open(path, 'rb') as stream:
                sources[os.path.abspath(path)] = hashlib.sha256(stream.read()).hexdigest()
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        self.emit('configuration', argv=sys.argv, cwd=os.getcwd(),
                  git_head=git.stdout.strip(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, source_sha256=sources,
                  settings=settings, model_min=self.minimum, model_max=self.maximum,
                  rejection_enabled=getattr(self.model, 'use_outlier_rejection', None),
                  model=str(self.model),
                  loss_formula='mean_per_image(log(1 + abs(pred - clamped_target)))')
        for split, dataset in datasets.items():
            if dataset is None:
                continue
            pairs = []
            for i in np.linspace(0, len(dataset) - 1, min(5, len(dataset)), dtype=int):
                paths = self.paths(dataset, int(i))
                raw = {}
                for name, path in paths.items():
                    if path.lower().endswith('.png') and os.path.isfile(path):
                        with Image.open(path) as image:
                            values = np.asarray(image)
                            raw[name] = dict(mode=image.mode, statistics=statistics(values))
                            if name != 'image_paths':
                                raw[name]['positive_depth_m'] = statistics(values[values > 0] / 256.)
                pairs.append(dict(index=int(i), paths=paths, raw_png=raw,
                                  exists={k: os.path.isfile(v) for k, v in paths.items()}))
            mismatches = []
            mismatch_count = 0
            for index in range(len(dataset)):
                paths = self.paths(dataset, index)
                identities = []
                for name, path in paths.items():
                    if name == 'intrinsics_paths':
                        continue
                    drive = re.search(r'\d{4}_\d{2}_\d{2}_drive_\d+_sync', path)
                    camera = re.search(r'image_0[23]', path)
                    identities.append((os.path.basename(path),
                                       drive.group() if drive else None,
                                       camera.group() if camera else None))
                mismatch = any(len(set(identity[k] for identity in identities
                                       if identity[k] is not None)) > 1 for k in range(3))
                if mismatch:
                    mismatch_count += 1
                    if len(mismatches) < 10:
                        mismatches.append(dict(index=index, paths=paths))
            self.emit('dataset', split=split, count=len(dataset), examples=pairs,
                      filename_or_kitti_identity_mismatch_count=mismatch_count,
                      mismatch_examples=mismatches,
                      crop=getattr(dataset, 'random_crop_shape', None),
                      crop_type=getattr(dataset, 'random_crop_type', None))
        print('[depth-debug] Diagnostics:', self.path)

    @staticmethod
    def paths(dataset, index):
        result = {}
        for name in ['image_paths', 'sparse_depth_paths', 'ground_truth_paths', 'intrinsics_paths']:
            paths = getattr(dataset, name, None)
            if paths is not None and paths[index] is not None:
                result[name] = paths[index]
        return result

    def begin(self, phase, step, sample=None, active=True):
        self.active = active
        self.context = dict(phase=phase, step=int(step), sample=sample)

    def register_hooks(self):
        def hook(name):
            def observe(module, inputs, output):
                if not self.active:
                    return
                with torch.no_grad():
                    if name == 'encoder':
                        self.emit('encoder', levels=[statistics(x) for x in output])
                    elif name.startswith('decoder.'):
                        feature, depth, logits, rejection, keep = output
                        weights = torch.sigmoid(logits.detach())
                        sparse = functional.interpolate(
                            inputs[3], size=depth.shape[-2:], mode='nearest')
                        self.emit('decoder', level=name, device=str(depth.device),
                                  metric_depth=statistics(depth),
                                  feature=statistics(feature),
                                  preclamp_below_min=float((depth < self.minimum).float().mean()),
                                  preclamp_above_max=float((depth > self.maximum).float().mean()),
                                  outgoing_weight_mean=float(weights.mean()),
                                  center_outgoing_weight=float(weights[:, 4].mean()),
                                  max_outgoing_weight=float(weights.max(1)[0].mean()),
                                  rejection_logits=statistics(rejection),
                                  keep_fraction=float(keep.float().mean()),
                                  keep_fraction_at_sparse=float(
                                      keep.sum() / (sparse > 0).sum().clamp_min(1)))
                    else:
                        self.emit('activation', name=name, values=statistics(output))
                        if name == 'model_output' and output.requires_grad:
                            output.register_hook(self.output_gradient)
            return observe
        for name, module in self.model.named_modules():
            if name in ['encoder', 'coarse_scaleless_depth', 'propagation_map',
                        'rejection_map', ''] or (name.startswith('decoder.') and name.count('.') == 1):
                self.handles.append(module.register_forward_hook(hook(name or 'model_output')))

    def output_gradient(self, gradient):
        self.emit('output_gradient', values=statistics(gradient))

    def inputs(self, name, image, sparse, target=None):
        self.emit('inputs', name=name, image=statistics(image), sparse=statistics(sparse),
                  sparse_valid_values=statistics(sparse[sparse > 0]),
                  sparse_density=float((sparse > 0).float().mean()),
                  target=None if target is None else statistics(target),
                  target_valid_values=None if target is None else statistics(target[target > 0]))
        if name == 'network_input':
            self.training_shape = list(image.shape[-2:])
        if target is not None and sparse.shape == target.shape:
            s, t = array(sparse), array(target)
            self.emit('sparse_vs_target', name=name,
                      errors=errors(s, t, (t > 0) & (s > 0)))

    def sparse_levels(self, sparse):
        '''Report the same direct nearest sampling used by the model.'''
        with torch.no_grad():
            h, w = sparse.shape[-2:]
            for level in range(5):
                size = ((h + 2 ** level - 1) // 2 ** level,
                        (w + 2 ** level - 1) // 2 ** level)
                sampled = functional.interpolate(sparse, size=size, mode='nearest')
                self.emit('sparse_sampling', level=level, size=size,
                          occupied_fraction=float((sampled > 0).float().mean()),
                          measurement_count=int((sampled > 0).sum()),
                          images_without_measurements=int(
                              ((sampled > 0).flatten(1).sum(1) == 0).sum()),
                          depth=statistics(sampled[sampled > 0]))

    def training_prediction(self, prediction, target, sparse, loss_info):
        if target is None:
            return
        p, t, s = array(prediction), array(target), array(sparse)
        valid = t > 0
        clipped = np.clip(t, self.minimum, self.maximum)
        self.emit('training_prediction', output=statistics(p),
                  per_image_spatial_std=[float(x.std()) for x in p],
                  target_below_min=int((valid & (t < self.minimum)).sum()),
                  target_above_max=int((valid & (t > self.maximum)).sum()),
                  raw_target=errors(p, t, valid), clamped_target=errors(p, clipped, valid),
                  at_sparse=errors(p, t, valid & (s > 0)),
                  away_from_sparse=errors(p, t, valid & (s <= 0)),
                  loss_info={k: float(v.detach()) for k, v in loss_info.items()
                             if torch.is_tensor(v) and v.numel() == 1},
                  clamp_min_fraction=float(np.mean(p <= self.minimum)),
                  clamp_max_fraction=float(np.mean(p >= self.maximum)))

    def before_step(self, optimizer):
        groups = {}
        self.before = {}
        for name, parameter in self.model.named_parameters():
            group = '.'.join(name.split('.')[:2])
            item = groups.setdefault(group, dict(grad_squared=0., weight_squared=0.,
                                                 no_gradient=0, nonfinite_gradient=0))
            item['weight_squared'] += float(parameter.detach().double().square().sum())
            if parameter.grad is None:
                item['no_gradient'] += 1
                continue
            gradient = parameter.grad.detach()
            item['nonfinite_gradient'] += int((~torch.isfinite(gradient)).sum())
            item['grad_squared'] += float(gradient.double().square().sum())
            self.before[name] = parameter.detach().cpu().clone()
        for item in groups.values():
            item['grad_norm'] = float(np.sqrt(item['grad_squared']))
            item['weight_norm'] = float(np.sqrt(item['weight_squared']))
        self.emit('gradients_before_optimizer', groups=groups,
                  learning_rates=[g['lr'] for g in optimizer.param_groups],
                  global_grad_norm=float(np.sqrt(sum(g['grad_squared'] for g in groups.values()))))

    def after_step(self):
        groups = {}
        for name, parameter in self.model.named_parameters():
            if name not in self.before:
                continue
            old = self.before[name].double()
            new = parameter.detach().cpu().double()
            group = '.'.join(name.split('.')[:2])
            item = groups.setdefault(group, dict(update_squared=0., weight_squared=0., nonfinite=0))
            item['update_squared'] += float((new - old).square().sum())
            item['weight_squared'] += float(old.square().sum())
            item['nonfinite'] += int((~torch.isfinite(new)).sum())
        for item in groups.values():
            item['relative_update'] = float(np.sqrt(item['update_squared']) /
                                            (np.sqrt(item['weight_squared']) + 1e-30))
        self.emit('optimizer_updates', groups=groups)
        self.before = {}

    def start_validation(self, step):
        self.begin('validation', step, active=False)
        self.worst = []
        self.buckets = {}
        self.inverse_failures = 0
        self.dump_count = 0

    def validation_sample(self, index, image, sparse, target, prediction,
                          minimum, maximum, dataset):
        p, t, s = [array(x).squeeze() for x in (prediction, target, sparse)]
        valid = (t > 0) & (t > minimum) & (t < maximum)
        groups = {'all': valid, 'at_sparse': valid & (s > 0),
                  'away_from_sparse': valid & (s <= 0)}
        for low, high in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]:
            groups['depth_{}_{}m'.format(low, high)] = valid & (t >= low) & (t < high)
        with torch.no_grad():
            near = functional.max_pool2d((sparse > 0).float(), 17, stride=1, padding=8)
        near = array(near).squeeze() > 0
        groups['within_8px_of_sparse'] = valid & near & (s <= 0)
        groups['beyond_8px_of_sparse'] = valid & ~near
        for band in range(4):
            mask = np.zeros_like(valid)
            mask[band * p.shape[0] // 4:(band + 1) * p.shape[0] // 4] = True
            groups['height_quarter_{}'.format(band)] = valid & mask
        if self.training_shape is not None:
            h, w = self.training_shape
            crop = np.zeros_like(valid)
            x = max(0, (p.shape[1] - w) // 2)
            crop[max(0, p.shape[0] - h):, x:x + w] = True
            groups['bottom_center_training_sized_crop'] = valid & crop
            groups['outside_training_sized_crop'] = valid & ~crop
        reports = {name: errors(p, t, mask) for name, mask in groups.items()}
        for name, report in reports.items():
            bucket = self.buckets.setdefault(name, dict(count=0, abs_sum_m=0., squared_sum_m2=0.))
            for field in bucket:
                bucket[field] += report.get('finite_count' if field == 'count' else field, 0)
        # Reproduce the actual dtype and reciprocal operations without altering metrics.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            inv_p = 1.0 / (0.001 * p[valid])
            inv_t = 1.0 / (0.001 * t[valid])
            inverse = inv_p - inv_t
            native_imae = np.mean(np.abs(inverse)) if inverse.size else np.nan
            native_irmse = np.sqrt(np.mean(inverse ** 2)) if inverse.size else np.nan
            bad = ~np.isfinite(inverse) | ~np.isfinite(inverse ** 2)
        self.context = dict(phase='validation', step=self.context['step'], sample=int(index))
        paths = self.paths(dataset, index)
        report = reports['all']
        self.emit('validation_sample', paths=paths, prediction=statistics(p),
                  valid_target=statistics(t[valid]), prediction_on_eval_mask=statistics(p[valid]),
                  buckets=reports, sparse_vs_target=errors(s, t, valid & (s > 0)),
                  prediction_vs_sparse=errors(p, s, s > 0),
                  native_imae=native_imae, native_irmse=native_irmse,
                  prediction_below_configured_min=int(np.sum(p < self.minimum)),
                  prediction_above_configured_max=int(np.sum(p > self.maximum)),
                  clamp_min_fraction=float(np.mean(p <= self.minimum)),
                  clamp_max_fraction=float(np.mean(p >= self.maximum)))
        if not np.isfinite(native_imae) or not np.isfinite(native_irmse):
            self.inverse_failures += 1
            coords = np.argwhere(valid)[bad][:20]
            self.emit('inverse_failure', paths=paths, coordinates_yx=coords.tolist(),
                      prediction_values=[float(p[tuple(c)]) for c in coords],
                      target_values=[float(t[tuple(c)]) for c in coords],
                      native_imae=native_imae, native_irmse=native_irmse,
                      float64_metrics=report)
            if self.dump_count < self.max_samples:
                self.save_arrays('inverse_failure_{}'.format(self.dump_count),
                                 image, s, t, p, valid, paths)
                self.dump_count += 1
        error_map = np.where(valid & np.isfinite(p) & np.isfinite(t), np.abs(p - t), -1)
        n_worst = min(10, int(valid.sum()))
        if n_worst:
            flat = np.argpartition(error_map.ravel(), error_map.size - n_worst)[-n_worst:]
            coords = np.column_stack(np.unravel_index(flat, p.shape))
            self.emit('worst_pixels', paths=paths, coordinates_yx=coords.tolist(),
                      prediction_m=[float(p[tuple(c)]) for c in coords],
                      target_m=[float(t[tuple(c)]) for c in coords],
                      sparse_m=[float(s[tuple(c)]) for c in coords])
        score = report.get('rmse_mm', float('inf'))
        if not np.isfinite(score):
            score = float('inf')
        if self.max_samples and (len(self.worst) < self.max_samples or score > self.worst[-1][0]):
            payload = dict(image=array(image).copy(), sparse=s.copy(), target=t.copy(),
                           prediction=p.copy(), mask=valid.copy(), paths=paths)
            self.worst.append((score, int(index), payload))
            self.worst.sort(key=lambda item: item[0], reverse=True)
            self.worst = self.worst[:self.max_samples]

    def save_arrays(self, name, image, sparse, target, prediction, mask, paths):
        # Fixed filenames bound disk usage; these are the latest validation examples.
        np.savez_compressed(os.path.join(self.directory, name + '.npz'),
                            image=array(image), sparse=sparse, target=target,
                            prediction=prediction, evaluation_mask=mask,
                            paths_json=json.dumps(paths), step=self.context['step'])

    def finish_validation(self):
        for bucket in self.buckets.values():
            count = max(1, bucket['count'])
            bucket['mae_mm'] = 1000 * bucket['abs_sum_m'] / count
            bucket['rmse_mm'] = 1000 * np.sqrt(bucket['squared_sum_m2'] / count)
        worst = []
        for rank, (score, index, data) in enumerate(self.worst):
            self.save_arrays('worst_{}'.format(rank), **data)
            worst.append(dict(index=index, rmse_mm=score, paths=data['paths']))
        summary = dict(step=self.context['step'], pixel_weighted_buckets=self.buckets,
                       inverse_failure_samples=self.inverse_failures, worst_samples=worst)
        self.emit('validation_summary', **summary)
        with open(os.path.join(self.directory, 'validation_latest.json'), 'w') as stream:
            json.dump(serializable(summary), stream, indent=2, allow_nan=False)
        print('[depth-debug] step={} inverse failures={} worst indices={} (see {})'.format(
            self.context['step'], self.inverse_failures,
            [entry['index'] for entry in worst], self.directory))
        self.worst = []
        self.active = False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
