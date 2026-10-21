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
from depth_completion_debug_analysis import StageDiagnostics, boundary_mask


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
        error_p999_m=float(np.percentile(absolute, 99.9)),
        fraction_over_0_1m=float(np.mean(absolute > 0.1)),
        fraction_over_0_5m=float(np.mean(absolute > 0.5)),
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


class DepthCompletionDebug(StageDiagnostics):
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
        self.rejection_observations = {}
        self.refinement_counts = {}
        self.rejection_reports = {}
        self.initialize_analysis()
        self.register_hooks()
        self.register_stage_hooks()

    def emit(self, event, **values):
        if self.probing or event in ['activation', 'encoder']:
            return
        with self.lock:
            record = dict(time=time.time(), event=event, schema_version=2,
                          run_id=self.run_id, elapsed_seconds=time.monotonic() - self.started_at,
                          images_seen_since_start=self.images_seen,
                          micro_batches_since_start=self.micro_batches_seen, **self.context)
            record.update(values)
            with open(self.path, 'a') as stream:
                stream.write(json.dumps(serializable(record), allow_nan=False) + '\n')

    def provenance(self, settings, datasets):
        self.training_shape = settings.get('train_shape')
        sources = {}
        root = os.path.dirname(__file__)
        for path in [inspect.getfile(type(self.model)), __file__,
                     inspect.getfile(StageDiagnostics),
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
        self.save_provenance(sources, settings, datasets)
        self.emit('configuration', argv=sys.argv, cwd=os.getcwd(),
                  git_head=git.stdout.strip(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, source_sha256=sources,
                  settings=settings, model_min=self.minimum, model_max=self.maximum,
                  rejection_enabled=getattr(self.model, 'use_outlier_rejection', None),
                  propagation_iterations=getattr(self.model, 'n_propagation_iterations', None),
                  model=str(self.model),
                  loss_definition='See captured loss_utils.py and model wrapper sources; detach sites in experiment_manifest',
                  propagation_activation='sigmoid',
                  sparse_scale_coordinates='original_selected_pixel_align_corners_true',
                  sparse_injection='direct_scale_injection_once_without_anchor_restoration')
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
        self.rejection_observations = {}
        self.refinement_counts = {}
        self.trace = {}
        self.anchor_context = {}

    def register_hooks(self):
        def hook(name):
            def observe(module, inputs, output):
                if not self.active:
                    return
                with torch.no_grad():
                    if name == 'encoder':
                        return
                    elif name.startswith('decoder.') and name.count('.') == 1:
                        feature, geometry, scale, logits, rejection, keep = output
                        depth = geometry * scale
                        weights = torch.sigmoid(logits.detach())
                        sparse = self.sparse_at_resolution(inputs[4], depth.shape[-2:])
                        self.emit('decoder', level=name, device=str(depth.device),
                                  propagation_iterations=module.n_propagation_iterations,
                                  metric_depth=statistics(depth),
                                  scaleless_depth=statistics(geometry),
                                  scale_map=statistics(scale),
                                  scaleless_mean_per_image=geometry.mean(dim=[1, 2, 3]).cpu().tolist(),
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
                        value = output['depth'] if isinstance(output, dict) else output
                        if name == 'scale_update' or name.endswith('.scale_update'):
                            scale = inputs[0][:, -4:-3]
                            accepted_scale = inputs[0][:, -3:-2]
                            keep = inputs[0][:, -2:-1]
                            refined = scale * torch.exp(torch.tanh(value))
                            level = 'bottom' if name == 'scale_update' else name.rsplit('.', 1)[0]
                            key = (name, str(value.device))
                            iteration = self.refinement_counts.get(key, 0) + 1
                            self.refinement_counts[key] = iteration
                            self.capture_stage(level, 'refinement_{}'.format(iteration),
                                               inputs[0][:, -5:-4], refined,
                                               multiplier=torch.exp(torch.tanh(value)),
                                               **self.anchor_context.get(level, {}))
                            self.emit('scale_refinement', level=level,
                                      refinement_iteration=iteration,
                                      scale_before_refinement=statistics(scale),
                                      scale_after_refinement=statistics(refined),
                                      accepted_scale_residual_before=statistics(
                                          (scale - accepted_scale)[keep > 0.0]),
                                      accepted_scale_residual=statistics(
                                          (refined - accepted_scale)[keep > 0.0]),
                                      restores_anchors=False)
                        if name == 'model_output':
                            if value.requires_grad:
                                value.register_hook(self.output_gradient)
                            if isinstance(output, dict):
                                metric = output['metric_depths_for_loss'][0]
                                if metric.requires_grad:
                                    metric.register_hook(self.output_gradient)
            return observe
        for name, module in self.model.named_modules():
            if name in ['encoder', 'coarse_scaleless_depth', 'scale_update', 'propagation_map',
                        'rejection_map', ''] or (name.startswith('decoder.') and name.count('.') == 1):
                self.handles.append(module.register_forward_hook(hook(name or 'model_output')))
            if name == 'scale_propagation' or name.endswith('.scale_propagation'):
                level = 'bottom' if name == 'scale_propagation' else name.rsplit('.', 1)[0]
                self.handles.append(module.register_forward_hook(self.scale_hook(level)))
            if name.endswith('.scale_update'):
                self.handles.append(module.register_forward_hook(hook(name)))
            if name == 'rejection_map' or name.endswith('.rejection_map'):
                level = 'bottom' if name == 'rejection_map' else name.rsplit('.', 1)[0]
                parent = self.model if level == 'bottom' else self.model.decoder[int(level.split('.')[1])]
                self.handles.append(module.register_forward_hook(self.rejection_hook(level, parent)))

    def scale_hook(self, level):
        '''Report direct scale injection and diffusion before learned refinement.'''
        def observe(module, inputs, output):
            if not self.active:
                return
            with torch.no_grad():
                scale, geometry, sparse, keep, logits, target = inputs
                propagated_scale = output
                accepted = keep > 0.0
                weights = torch.sigmoid(logits)
                if self.context.get('phase') == 'validation':
                    self.anchor_context[level] = dict(keep=keep)
                    self.capture_stage(level, 'initial_global_scale' if level == 'bottom' else 'after_geometry_update',
                                       geometry, scale)
                    self.capture_stage(level, 'after_propagation', geometry, propagated_scale,
                                       sparse_ratio=target,
                                       outgoing_weights=weights, **self.anchor_context[level])
                self.emit('scale_propagation', level=level,
                          propagation_iterations=module.n_propagation_iterations,
                          output_stage='before_refinement',
                          inherited_scale=statistics(scale),
                          injected_scale=statistics(keep * target + (1.0 - keep) * scale),
                          propagated_scale=statistics(propagated_scale),
                          sparse_scale_target=statistics(target[accepted]),
                          accepted_scale_residual=statistics((propagated_scale - target)[accepted]),
                          accepted_scale_residual_scope='propagated_scale_before_refinement',
                          sparse_scale_coordinates='original_selected_pixel_align_corners_true',
                          scaleless_depth=statistics(geometry),
                          outgoing_weight_sum=statistics(weights.sum(dim=1)),
                          center_outgoing_weight=statistics(weights[:, 4]),
                          propagation_activation='sigmoid',
                          restores_anchors=False)
        return observe

    def rejection_hook(self, level, parent):
        '''Report both class scores after the head's learned bias is added.'''
        def observe(module, inputs, output):
            if not self.active:
                return
            with torch.no_grad():
                logits = output + parent.rejection_bias.to(output.device)
                margin = logits[:, 0:1] - logits[:, 1:2]
                probability = functional.softmax(logits, dim=1)[:, 0:1]
                # Rejection input ends in scaled sparse depth; positivity still
                # identifies the same selected measurements as the metric map.
                measured = inputs[0][:, -1:] > 0.0
                self.emit('rejection', level=level,
                          measurement_count=int(measured.sum()),
                          keep_minus_reject_at_sparse=statistics(margin[measured]),
                          keep_probability_at_sparse=statistics(probability[measured]),
                          deterministic_keep_fraction_at_sparse=float(
                              ((margin >= 0.0) & measured).sum() / measured.sum().clamp_min(1)),
                          depth_context_input=statistics(inputs[0][:, -2:-1]),
                          scaled_sparse_input=statistics(inputs[0][:, -1:]))
                # Validation uses argmax, so this is the actual keep decision.
                # Training uses sampled Gumbel decisions; do not label those
                # using untransformed ground truth from the training batch.
                if self.context['phase'] == 'validation':
                    self.rejection_observations[level] = dict(
                        size=tuple(margin.shape[-2:]),
                        logits=logits.detach(),
                        keep=(margin >= 0.0).detach())
        return observe

    def rejection_quality(self, sparse, target):
        '''Compare selected measurements with GT at their original coordinates.

        These categories describe disagreement with the supplied ground truth;
        they are diagnostics, not outlier labels or additional training losses.
        Missing GT is excluded. No depth/validity averaging is used.
        '''
        sampler = getattr(sys.modules[type(self.model).__module__],
                          'sparse_depth_at_resolution', None)
        if sampler is None:
            return
        with torch.no_grad():
            for level, observation in self.rejection_observations.items():
                sampled, indices = sampler(sparse, observation['size'], return_indices=True)
                selected_target = torch.gather(
                    target.flatten(2), 2, indices.flatten(2)).reshape_as(sampled)
                known = (sampled > 0.0) & (selected_target > 0.0) & torch.isfinite(selected_target)
                difference = torch.abs(sampled - selected_target)
                keep = observation['keep'].to(sparse.device)
                categories = {
                    'all_with_ground_truth': known,
                    'accurate_within_0_1m': known & (difference <= 0.1),
                    'disagreement_over_3m': known & (difference > 3.0),
                    'disagreement_over_5percent': known & (
                        difference / selected_target.clamp_min(1e-7) > 0.05)
                }
                reports = {}
                for name, mask in categories.items():
                    count = int(mask.sum())
                    rejected = int((mask & ~keep).sum())
                    reports[name] = dict(
                        count=count, kept=count - rejected, rejected=rejected,
                        rejection_fraction=None if count == 0 else rejected / count)
                self.emit('rejection_quality', level=level,
                          ground_truth_scope='original_selected_pixel',
                          sparse_vs_target=errors(sampled, selected_target, array(known)),
                          categories=reports)
                self.supervised_rejection_report(level, observation['logits'], sampled, selected_target)

    def output_gradient(self, gradient):
        if not self.probing:
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
        if name == 'validation_input' and target is not None:
            self.rejection_quality(sparse, target)

    def sparse_at_resolution(self, sparse, output_size):
        '''Use the model's sparse selector when reporting measurements.'''
        sampler = getattr(sys.modules[type(self.model).__module__],
                          'sparse_depth_at_resolution', None)
        if sampler is not None:
            return sampler(sparse, output_size)
        return functional.interpolate(sparse, size=output_size, mode='nearest')

    def sparse_levels(self, sparse):
        '''Report sparse coverage using the same selection as the model.'''
        with torch.no_grad():
            h, w = sparse.shape[-2:]
            for level in range(5):
                size = ((h + 2 ** level - 1) // 2 ** level,
                        (w + 2 ** level - 1) // 2 ** level)
                sampled = self.sparse_at_resolution(sparse, size)
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
        rejection_heads = {}
        self.before = {}
        for name, parameter in self.model.named_parameters():
            group = '.'.join(name.split('.')[:2])
            item = groups.setdefault(group, dict(grad_squared=0., weight_squared=0.,
                                                 no_gradient=0, nonfinite_gradient=0))
            # Separate each rejection head from its decoder's other parameters.
            head_name = None
            if name == 'rejection_bias' or name.startswith('rejection_map.'):
                head_name = 'bottom'
            elif '.rejection_map.' in name or name.endswith('.rejection_bias'):
                head_name = name.split('.rejection_')[0]
            head = None if head_name is None else rejection_heads.setdefault(
                head_name, dict(grad_squared=0., weight_squared=0.,
                                no_gradient=0, nonfinite_gradient=0))
            weight_squared = float(parameter.detach().double().square().sum())
            item['weight_squared'] += weight_squared
            if head is not None:
                head['weight_squared'] += weight_squared
            if parameter.grad is None:
                item['no_gradient'] += 1
                if head is not None:
                    head['no_gradient'] += 1
                continue
            gradient = parameter.grad.detach()
            nonfinite = int((~torch.isfinite(gradient)).sum())
            grad_squared = float(gradient.double().square().sum())
            item['nonfinite_gradient'] += nonfinite
            item['grad_squared'] += grad_squared
            if head is not None:
                head['nonfinite_gradient'] += nonfinite
                head['grad_squared'] += grad_squared
            self.before[name] = parameter.detach().cpu().clone()
        for item in list(groups.values()) + list(rejection_heads.values()):
            item['grad_norm'] = float(np.sqrt(item['grad_squared']))
            item['weight_norm'] = float(np.sqrt(item['weight_squared']))
        prediction_heads = {}
        for name, parameter in self.model.named_parameters():
            if not any(part in ['coarse_scaleless_depth', 'depth_update',
                                'scale_update', 'propagation_map']
                       for part in name.split('.')):
                continue
            key = name.split('.conv.')[0]
            item = prediction_heads.setdefault(
                key, dict(grad_squared=0., no_gradient=0, nonfinite_gradient=0))
            if parameter.grad is None:
                item['no_gradient'] += 1
            else:
                gradient = parameter.grad.detach()
                item['grad_squared'] += float(gradient.double().square().sum())
                item['nonfinite_gradient'] += int((~torch.isfinite(gradient)).sum())
        for item in prediction_heads.values():
            item['grad_norm'] = float(np.sqrt(item['grad_squared']))
        self.emit('gradients_before_optimizer', groups=groups, rejection_heads=rejection_heads,
                  prediction_heads=prediction_heads,
                  optimizer=type(optimizer).__name__,
                  optimizer_settings=[{k: v for k, v in group.items() if k != 'params'}
                                      for group in optimizer.param_groups],
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
        self.emit('optimizer_updates', groups=groups, optimizer_step_invoked=True,
                  no_parameter_change=not any(g['update_squared'] > 0 for g in groups.values()),
                  scope='observed invoked optimizer step; unchanged parameters are not automatically a skipped step')
        self.before = {}

    def start_validation(self, step):
        self.begin('validation', step, active=False)
        self.worst = []
        self.buckets = {}
        self.inverse_failures = 0
        self.dump_count = 0
        self.validation_number += 1
        self.stage_reports = []
        self.rejection_reports = {}

    def validation_sample(self, index, image, sparse, target, prediction,
                          minimum, maximum, dataset):
        p, t, s = [array(x).squeeze() for x in (prediction, target, sparse)]
        valid = (t > 0) & (t > minimum) & (t < maximum)
        groups = {'all': valid, 'at_sparse': valid & (s > 0),
                  'away_from_sparse': valid & (s <= 0)}
        edge = boundary_mask(t, valid)
        groups['near_gt_boundary'] = valid & edge
        groups['away_from_gt_boundary'] = valid & ~edge
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
        # All images contribute to aggregate buckets. Detailed distributions
        # are retained for fixed samples only, replacing repetitive large logs.
        self.emit('validation_sample', paths=paths,
                  buckets=reports if self.active else {'all': report},
                  native_imae=native_imae, native_irmse=native_irmse,
                  prediction_below_configured_min=int(np.sum(p < self.minimum)),
                  prediction_above_configured_max=int(np.sum(p > self.maximum)),
                  clamp_min_fraction=float(np.mean(p <= self.minimum)),
                  clamp_max_fraction=float(np.mean(p >= self.maximum)))
        if self.active and self.trace:
            self.analyze_stages(array(image), s, t, p, valid, paths, 'fixed_{}'.format(index))
            self.track_fixed_errors(index, p, t, valid, paths)
            if self.validation_number == 1 or self.context['step'] % self.probe_interval == 0:
                self.probe_anchor_messages(image, sparse, t, p, valid)
        self.trace = {}
        self.anchor_context = {}
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
        if n_worst and self.active:
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
            # A small eval replay provides internal maps for the actual worst
            # samples without retaining internal maps for the whole dataset.
            if not self.model.training:
                self.begin('validation', self.context['step'], sample=index)
                device = next(self.model.parameters()).device
                with torch.random.fork_rng(), torch.no_grad():
                    self.model(torch.as_tensor(data['image'], device=device),
                               torch.as_tensor(data['sparse'], device=device).reshape(1, 1, *data['sparse'].shape))
                self.rejection_quality(
                    torch.as_tensor(data['sparse'], device=device).reshape(1, 1, *data['sparse'].shape),
                    torch.as_tensor(data['target'], device=device).reshape(1, 1, *data['target'].shape))
                self.analyze_stages(data['image'], data['sparse'], data['target'], data['prediction'],
                                    data['mask'], data['paths'], 'worst_{}_stages'.format(rank))
        summary = dict(step=self.context['step'], pixel_weighted_buckets=self.buckets,
                       inverse_failure_samples=self.inverse_failures, worst_samples=worst,
                       rejection_supervision=self.rejection_summary(),
                       rejection_summary_scope='fixed and worst samples, deduplicated per image/level')
        self.emit('validation_summary', **summary)
        with open(os.path.join(self.directory, 'validation_latest.json'), 'w') as stream:
            json.dump(serializable(summary), stream, indent=2, allow_nan=False)
        with open(os.path.join(self.directory, 'stages_latest.json'), 'w') as stream:
            json.dump(serializable(dict(step=self.context['step'], run_id=self.run_id,
                                        records=self.stage_reports)), stream, indent=2, allow_nan=False)
        print('[depth-debug] step={} inverse failures={} worst indices={} (see {})'.format(
            self.context['step'], self.inverse_failures,
            [entry['index'] for entry in worst], self.directory))
        self.worst = []
        self.active = False
        self.trace = {}
        self.anchor_context = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
