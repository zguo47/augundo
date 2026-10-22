'''Read-only, sampled diagnostics for geometry, scale, and anchor influence.

All stage arrays are detached CPU copies. Ground truth is used only to assess
the recorded predictions, never as an input to the model or an optimizer step.
Native maps are saved; accuracy uses bilinearly resized native metric products
on one common full-resolution evaluation mask, matching auxiliary supervision.
'''

import hashlib
import inspect
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import ndimage


def resize_map(value, size):
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(1, 1, *value.shape[-2:])
    return functional.interpolate(tensor, size=size, mode='bilinear', align_corners=True).numpy()[0, 0]


def anchor_distance(mask):
    # Euclidean distance, in the coordinates of this mask. Empty source sets
    # have infinite distance, not distance to an implicit image border.
    return ndimage.distance_transform_edt(~mask) if mask.any() else np.full(mask.shape, np.inf)


def boundary_mask(target, valid):
    boundary = np.zeros_like(valid)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        difference = np.abs(target[left] - target[right])
        edge = valid[left] & valid[right] & (difference > 0.1) & (
            difference > 0.05 * np.minimum(target[left], target[right]))
        boundary[left] |= edge
        boundary[right] |= edge
    return ndimage.binary_dilation(boundary, iterations=2)


class StageDiagnostics(object):
    '''Mixin keeping expensive analysis separate from routine training logging.'''

    def initialize_analysis(self):
        self.started_at = time.monotonic()
        self.run_id = str(time.time_ns())
        self.probe_interval = 10 * self.interval
        self.images_seen = 0
        self.micro_batches_seen = 0
        self.trace = {}
        self.fixed_indices = set(range(self.max_samples))
        self.previous_fixed = {}
        self.validation_number = 0
        self.probing = False
        self.settings = {}
        self.stage_reports = []

    def record_training_batch(self, batch_size, epoch, micro_step):
        self.images_seen += batch_size
        self.micro_batches_seen += 1
        self.training_progress = dict(epoch=epoch, micro_step=micro_step)

    def save_provenance(self, sources, settings, datasets):
        self.settings = settings
        validation = datasets.get('validation')
        if validation is not None:
            self.fixed_indices = set(np.linspace(
                0, max(0, len(validation) - 1), min(self.max_samples, len(validation)), dtype=int).tolist())
        destination = os.path.join(self.directory, 'sources', self.run_id)
        os.makedirs(destination, exist_ok=True)
        manifest = {}
        for index, path in enumerate(sources):
            with open(path, 'rb') as stream:
                content = stream.read()
            name = '{}_{}'.format(index, os.path.basename(path))
            with open(os.path.join(destination, name), 'wb') as stream:
                stream.write(content)
            manifest[path] = name
        # Capture replay-relevant state without drawing random numbers. This
        # records the debug startup state, not a complete dataloader checkpoint.
        torch.save(dict(torch_cpu=torch.get_rng_state(),
                        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        numpy=np.random.get_state(), python=random.getstate()),
                   os.path.join(destination, 'rng_start.pt'))
        model_source = inspect.getsource(type(self.model))
        detach_sites = []
        with open(inspect.getfile(type(self.model))) as stream:
            detach_sites = [dict(line=i, code=line.strip()) for i, line in enumerate(stream, 1)
                            if '.detach()' in line]
        identities = {}
        for split, dataset in datasets.items():
            if dataset is not None:
                digest = hashlib.sha256()
                for i in range(len(dataset)):
                    digest.update(json.dumps(self.paths(dataset, i), sort_keys=True).encode())
                identities[split] = digest.hexdigest()
        self.emit('experiment_manifest', source_directory=destination, source_files=manifest,
                  rng_state_file='rng_start.pt',
                  dataset_order_sha256=identities, detach_sites=detach_sites,
                  metric_geometry_detached=getattr(
                      self.model, 'detach_scaleless_depth', 'geometry.detach() * metric_scale' in model_source),
                  torch_initial_seed=torch.initial_seed(),
                  torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                  numpy_rng_sha256=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                  python_rng_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  fixed_sample_indices=sorted(self.fixed_indices),
                  probe_interval_optimizer_steps=self.probe_interval,
                  fixed_probe_policy='first validation, then steps divisible by probe interval',
                  stage_metric_policy='resize native Z*M; raw GT; validation mask; no clamp',
                  boundary_policy='valid adjacent GT: jump >0.1m AND >5%, dilated 2 pixels',
                  resume_paths=settings.get('restore_paths', []))

    def capture_stage(self, level, stage, geometry, scale, **extra):
        if not self.active or self.probing or self.context.get('phase') != 'validation':
            return
        # Validation forwards one image at a time. Capture only maps, not the
        # C-channel feature tensors, to bound retained memory and disk usage.
        with self.lock:
            entry = dict(level=level, stage=stage,
                         geometry=geometry.detach().float().cpu().numpy()[0, 0].copy(),
                         scale=scale.detach().float().cpu().numpy()[0, 0].copy())
            for name, value in extra.items():
                entry[name] = value.detach().cpu().numpy()[0].copy()
            self.trace.setdefault(level, []).append(entry)

    def register_stage_hooks(self):
        for index, block in enumerate(getattr(self.model, 'decoder', [])):
            level = 'decoder.{}'.format(index)
            def before(module, inputs, level=level):
                self.capture_stage(level, 'before_upsampling', inputs[2], inputs[3])
            def fused(module, inputs, level=level):
                # Fusion concatenates features, skip, Z, M, sparse depth.
                self.capture_stage(level, 'after_upsampling', inputs[0][:, -3:-2], inputs[0][:, -2:-1])
            self.handles.append(block.register_forward_pre_hook(before))
            self.handles.append(block.fusion.register_forward_pre_hook(fused))

    def gradient_objectives(self, loss_info):
        '''Extra VJPs do not write .grad or consume the subsequent backward graph.'''
        step = self.context['step']
        if step != 1 and step % self.probe_interval:
            return
        terms = [loss_info.get('loss_si'), loss_info.get('loss_log_l1')]
        if not all(torch.is_tensor(term) and term.requires_grad for term in terms):
            self.emit('gradient_objectives', unavailable='SI/metric graph terms not exposed')
            return
        names = ['si', 'metric']
        rejection_term = loss_info.get('loss_rejection_weighted')
        if torch.is_tensor(rejection_term) and rejection_term.requires_grad:
            names.append('rejection')
            terms.append(rejection_term)
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        self.probing = True
        try:
            vectors = [torch.autograd.grad(term, [p for _, p in named], retain_graph=True,
                                          allow_unused=True) for term in terms]
        finally:
            self.probing = False
        groups = {}
        for index, (name, _) in enumerate(named):
            group = '.'.join(name.split('.')[:2])
            item = groups.setdefault(group, {})
            gradients = [vector[index] for vector in vectors]
            for objective, gradient in zip(names, gradients):
                key = objective + '_squared'
                item[key] = item.get(key, 0.) + (0. if gradient is None else
                    float(gradient.detach().double().square().sum()))
                key = objective + '_unused'
                item[key] = item.get(key, 0) + int(gradient is None)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    key = names[i] + '_' + names[j] + '_dot'
                    item[key] = item.get(key, 0.) + (0. if gradients[i] is None or gradients[j] is None else
                        float((gradients[i].detach().double() * gradients[j].detach().double()).sum()))
        for item in groups.values():
            for objective in names:
                item[objective + '_norm'] = item.pop(objective + '_squared') ** 0.5
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    denominator = item[names[i] + '_norm'] * item[names[j] + '_norm']
                    key = 'cosine' if (i, j) == (0, 1) else 'cosine_' + names[i] + '_' + names[j]
                    dot = item.pop(names[i] + '_' + names[j] + '_dot')
                    item[key] = dot / denominator if denominator else None
        self.emit('gradient_objectives', groups=groups,
                  scope='current microbatch before accumulation division; rejection includes its configured loss weight')

    def supervised_rejection_report(self, level, logits, selected, matched):
        '''Report the actual label policy separately from generic GT disagreements.'''
        from utils.src import loss_utils
        from depth_completion_debug import statistics
        keep_tolerance = self.settings.get('rejection_keep_tolerance', (0.1, 0.02))
        reject_tolerance = self.settings.get('rejection_reject_tolerance', (0.3, 0.05))
        with torch.no_grad():
            labels = loss_utils.rejection_supervision_targets(
                selected, matched, keep_tolerance, reject_tolerance)
            prediction = logits.argmax(1, keepdim=True)
            probability = logits.softmax(1)[:, 0:1]
            measured = (selected > 0) & torch.isfinite(selected)
            known = measured & (matched > 0) & torch.isfinite(matched)
            keep, reject = labels == 0, labels == 1
            counts = dict(true_keep=int((keep & (prediction == 0)).sum()),
                          false_reject=int((keep & (prediction == 1)).sum()),
                          true_reject=int((reject & (prediction == 1)).sum()),
                          false_keep=int((reject & (prediction == 0)).sum()))
            n_keep, n_reject = int(keep.sum()), int(reject.sum())
            n_predicted_reject = counts['true_reject'] + counts['false_reject']
            resolution = 4 if level == 'bottom' else 3 - int(level.split('.')[1])
            report = dict(level=level, keep_target_count=n_keep, reject_target_count=n_reject,
                          ambiguous_count=int((known & (labels == -1)).sum()),
                          missing_gt_count=int((measured & ~known).sum()),
                          confusion=counts,
                          reject_recall=counts['true_reject'] / n_reject if n_reject else None,
                          reject_precision=counts['true_reject'] / n_predicted_reject if n_predicted_reject else None,
                          keep_recall=counts['true_keep'] / n_keep if n_keep else None,
                          keep_probability_on_good=statistics(probability[keep]),
                          keep_probability_on_bad=statistics(probability[reject]),
                          keep_tolerance=keep_tolerance, reject_tolerance=reject_tolerance,
                          supervision_enabled=self.settings.get('rejection_supervision_active', False) and
                          resolution in self.settings.get('rejection_supervision_levels', range(5)),
                          rejection_loss_type=self.settings.get('rejection_loss_type', 'cross_entropy'),
                          rejection_loss_enabled=self.settings.get(
                              'rejection_loss_active', self.settings.get('rejection_supervision_active', False)) and
                          resolution in self.settings.get('rejection_supervision_levels', range(5)),
                          rejection_keep_weight=self.settings.get('rejection_keep_weight', 2.0),
                          resolution_weight=1.0 / (4 ** resolution),
                          balanced_ce=float(loss_utils.balanced_rejection_loss_func(
                              logits, labels, self.settings.get('rejection_keep_weight', 2.0))),
                          label_scope='selected original pixel in aligned input coordinates; raw unclamped GT')
            self.emit('rejection_supervision', **report)
            if self.context.get('phase') == 'validation':
                self.rejection_reports[(self.context.get('sample'), level)] = report
                for entry in self.trace.get(level, []):
                    if entry['stage'] == 'after_propagation':
                        entry['rejection_logits'] = logits.detach().cpu().numpy()[0].copy()
                        entry['rejection_labels'] = labels.cpu().numpy()[0].copy()

    def rejection_training(self, logits, sparse, target):
        '''The supplied target already underwent the input's geometric transform.'''
        import sys
        sampler = getattr(sys.modules[type(self.model).__module__], 'sparse_depth_at_resolution')
        for resolution, scores in enumerate(logits):
            selected, indices = sampler(sparse, scores.shape[-2:], return_indices=True)
            matched = torch.gather(target.flatten(2), 2, indices.flatten(2)).reshape_as(selected)
            level = 'bottom' if resolution == 4 else 'decoder.{}'.format(3 - resolution)
            self.supervised_rejection_report(level, scores, selected, matched)

    def rejection_summary(self):
        result = {}
        for (_, level), report in self.rejection_reports.items():
            bucket = result.setdefault(level, dict(samples=0, keep_target_count=0, reject_target_count=0,
                ambiguous_count=0, missing_gt_count=0,
                confusion=dict(true_keep=0, false_reject=0, true_reject=0, false_keep=0)))
            bucket['samples'] += 1
            for key in ['keep_target_count', 'reject_target_count', 'ambiguous_count', 'missing_gt_count']:
                bucket[key] += report[key]
            for key, value in report['confusion'].items():
                bucket['confusion'][key] += value
        for bucket in result.values():
            counts = bucket['confusion']
            predicted = counts['true_reject'] + counts['false_reject']
            bucket['reject_recall'] = counts['true_reject'] / bucket['reject_target_count'] if bucket['reject_target_count'] else None
            bucket['reject_precision'] = counts['true_reject'] / predicted if predicted else None
            bucket['keep_recall'] = counts['true_keep'] / bucket['keep_target_count'] if bucket['keep_target_count'] else None
        return result

    def analyze_stages(self, image, sparse, target, prediction, valid, paths, name):
        from depth_completion_debug import errors, statistics
        t = np.asarray(target).squeeze()
        s = np.asarray(sparse).squeeze()
        p = np.asarray(prediction).squeeze()
        maps = {}
        reports = []
        edge = boundary_mask(t, valid)
        for level, entries in self.trace.items():
            previous = None
            anchor_groups = None
            for entry in entries:
                z, m = entry['geometry'], entry['scale']
                metric = resize_map(z * m, t.shape)
                z_full, m_full = resize_map(z, t.shape), resize_map(m, t.shape)
                mask = valid & np.isfinite(metric) & np.isfinite(z_full) & (z_full > 0)
                # Fit exactly one multiplicative scalar per image, only for
                # diagnostic alignment. This never changes the model output.
                alignment = (float(np.exp(np.mean(np.log(t[mask].astype(np.float64)) -
                                                 np.log(z_full[mask].astype(np.float64)))))
                             if mask.any() else 1.)
                scale_target = t / np.maximum(z_full, 1e-7)
                log_ratio = np.log(np.maximum(m_full, 1e-7)) - np.log(np.maximum(scale_target, 1e-7))
                record = dict(level=level, stage=entry['stage'], native_size=list(z.shape),
                              sample=self.context.get('sample'), step=self.context['step'],
                              metric=errors(metric, t, valid),
                              aligned_geometry=errors(alignment * z_full, t, valid),
                              alignment=alignment,
                              scale_log_error=statistics(log_ratio[mask]),
                              factorization_product=errors(z_full * m_full, t, valid),
                              factorization_scope='separately resized Z and M; metric uses resized native product',
                              at_sparse=errors(metric, t, valid & (s > 0)),
                              away_from_sparse=errors(metric, t, valid & (s <= 0)),
                              near_gt_boundary=errors(metric, t, valid & edge),
                              away_from_gt_boundary=errors(metric, t, valid & ~edge))
                if previous is not None:
                    before_error, after_error = np.abs(previous - t), np.abs(metric - t)
                    compare = mask & np.isfinite(previous)
                    record['change'] = dict(
                        improved_fraction=float(np.mean(after_error[compare] < before_error[compare] - 1e-6)) if compare.any() else None,
                        worsened_fraction=float(np.mean(after_error[compare] > before_error[compare] + 1e-6)) if compare.any() else None,
                        mean_absolute_error_change_m=float(np.mean(after_error[compare] - before_error[compare])) if compare.any() else None)
                previous = metric
                if 'multiplier' in entry:
                    multiplier = entry['multiplier']
                    record['multiplier'] = statistics(multiplier)
                    record['near_update_bound_fraction'] = float(np.mean(np.abs(np.log(multiplier)) > .95))
                    record['near_identity_fraction'] = float(np.mean(np.abs(np.log(multiplier)) < .001))
                if 'keep' in entry:
                    if anchor_groups is None:
                        sampled, indices = self.sparse_indices(sparse, z.shape)
                        selected_target = t.ravel()[indices.ravel()].reshape(indices.shape)
                        keep = entry['keep'].squeeze(0) > 0
                        original = np.zeros(t.shape, dtype=bool)
                        original.ravel()[indices[keep]] = True
                        distance = anchor_distance(original)
                        known = (selected_target > 0) & np.isfinite(selected_target)
                        bad = keep & known & (np.abs(sampled - selected_target) > np.maximum(.1, .05 * selected_target))
                        bad_original = np.zeros(t.shape, dtype=bool)
                        bad_original.ravel()[indices[bad]] = True
                        anchor_groups = (sampled, indices, distance, anchor_distance(bad_original) <= 8)
                    sampled, indices, distance, near_bad = anchor_groups
                    record['accepted_original_pixel_count'] = int(np.count_nonzero(distance == 0))
                    record['accepted_anchor_distance_errors'] = {
                        '{}_{}px'.format(low, high): errors(metric, t, valid & (distance > low) & (distance <= high))
                        for low, high in [(-1, 0), (0, 4), (4, 8), (8, 16), (16, 32), (32, np.inf)]}
                    record['near_accepted_disagreeing_anchor'] = errors(metric, t, valid & near_bad)
                    record['bad_anchor_scope'] = 'spatial association only, not causal attribution; >max(0.1m,5% GT)'
                    maps[level + '__selected_indices'] = indices
                    maps[level + '__selected_sparse'] = sampled
                self.emit('stage_accuracy', **record)
                reports.append(record)
                for key, value in entry.items():
                    if isinstance(value, np.ndarray):
                        maps[level + '__' + entry['stage'] + '__' + key] = value
        self.stage_reports.extend(reports)
        np.savez_compressed(os.path.join(self.directory, name + '.npz'),
                            image=np.asarray(image), sparse=s, target=t, prediction=p,
                            evaluation_mask=valid, paths_json=json.dumps(paths),
                            step=self.context['step'], run_id=self.run_id, **maps)
        self.emit('stage_snapshot', name=name, stage_count=len(reports), paths=paths)

    def sparse_indices(self, sparse, size):
        import sys
        sampler = getattr(sys.modules[type(self.model).__module__], 'sparse_depth_at_resolution')
        tensor = torch.as_tensor(sparse).reshape(1, 1, *np.asarray(sparse).shape[-2:])
        selected, indices = sampler(tensor, size, return_indices=True)
        return selected.numpy()[0, 0], indices.numpy()[0, 0]

    def track_fixed_errors(self, index, prediction, target, valid, paths):
        p, t = np.asarray(prediction).squeeze(), np.asarray(target).squeeze()
        identity = json.dumps(paths, sort_keys=True)
        previous = self.previous_fixed.get(index)
        if previous is not None and previous['identity'] == identity and previous['shape'] == p.shape:
            ys, xs = previous['coordinates']
            valid_now = valid[ys, xs] & np.isfinite(p[ys, xs])
            self.emit('fixed_error_tracks', previous_step=previous['step'],
                      coordinates_yx=np.column_stack([ys, xs]).tolist(),
                      previous_abs_error_m=previous['error'].tolist(),
                      current_abs_error_m=np.abs(p[ys, xs] - t[ys, xs]).tolist(),
                      current_valid=valid_now.tolist())
        candidates = np.flatnonzero(valid & np.isfinite(p))
        if candidates.size:
            error = np.abs(p.ravel()[candidates] - t.ravel()[candidates])
            n = min(50, len(candidates))
            indices = candidates[np.argpartition(error, len(error) - n)[-n:]]
            coordinates = np.unravel_index(indices, p.shape)
            self.previous_fixed[index] = dict(identity=identity, shape=p.shape,
                coordinates=coordinates, error=np.abs(p[coordinates] - t[coordinates]), step=self.context['step'])

    def probe_anchor_messages(self, image, sparse, target, prediction, valid):
        '''Bypass direct scale injection and neighboring scale exchanges.

        Preserve the inherited prior, sparse encoder inputs, rejection, and
        direct K*r refinement feedback while bypassing the propagation module.
        The extra eval forward cannot update parameters or normalization state.
        '''
        from depth_completion_debug import errors, statistics
        if self.model.training:
            return
        handles = []
        for name, module in self.model.named_modules():
            if name == 'scale_propagation' or name.endswith('.scale_propagation'):
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output: inputs[0]))
        active = self.active
        self.active, self.probing = False, True
        device = next(self.model.parameters()).device
        try:
            with torch.random.fork_rng(), torch.no_grad():
                result = self.model(image.to(device), sparse.to(device))
            result = result.detach().cpu().numpy().squeeze()
        finally:
            for handle in handles:
                handle.remove()
            self.active, self.probing = active, False
        p, t = np.asarray(prediction).squeeze(), np.asarray(target).squeeze()
        self.emit('scale_propagation_ablation', enabled=errors(p, t, valid),
                  disabled=errors(result, t, valid), change=statistics((result - p)[valid]),
                  scope='all levels scale injection/diffusion bypassed; direct sparse inputs and bottom prior retained')
