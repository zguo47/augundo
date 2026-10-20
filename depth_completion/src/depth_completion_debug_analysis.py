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
                  metric_geometry_detached='geometry.detach() * metric_scale' in model_source,
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
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        self.probing = True
        try:
            vectors = [torch.autograd.grad(term, [p for _, p in named], retain_graph=True,
                                          allow_unused=True) for term in terms]
        finally:
            self.probing = False
        groups = {}
        for (name, _), si, metric in zip(named, *vectors):
            group = '.'.join(name.split('.')[:2])
            item = groups.setdefault(group, dict(si_squared=0., metric_squared=0., dot=0.,
                                                 si_unused=0, metric_unused=0))
            item['si_unused'] += int(si is None)
            item['metric_unused'] += int(metric is None)
            if si is not None:
                item['si_squared'] += float(si.detach().double().square().sum())
            if metric is not None:
                item['metric_squared'] += float(metric.detach().double().square().sum())
            if si is not None and metric is not None:
                item['dot'] += float((si.detach().double() * metric.detach().double()).sum())
        for item in groups.values():
            item['si_norm'] = item.pop('si_squared') ** 0.5
            item['metric_norm'] = item.pop('metric_squared') ** 0.5
            denominator = item['si_norm'] * item['metric_norm']
            item['cosine'] = item.pop('dot') / denominator if denominator else None
        self.emit('gradient_objectives', groups=groups,
                  scope='current microbatch; resolution-weighted objectives before accumulation division')

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
                if 'support' in entry:
                    support = resize_map(entry['support'], t.shape)
                    # Support is interpolated only for evaluation grouping; it
                    # is not a probability of correct surface association.
                    record['support_errors'] = {label: errors(metric, t, valid & selection)
                        for label, selection in {
                            'none_le_1e-7': support <= 1e-7,
                            'weak_1e-7_to_1e-3': (support > 1e-7) & (support <= 1e-3),
                            'strong_gt_1e-3': support > 1e-3}.items()}
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
        '''Disable only Q/support inputs, not the direct sparse constraints.

        This measures the model's use of propagated messages while preserving
        the prior, selected measurements, rejection, and direct K*r feedback.
        The extra eval forward cannot update parameters or normalization state.
        '''
        from depth_completion_debug import errors, statistics
        if self.model.training:
            return
        handles = []
        for name, module in self.model.named_modules():
            if name == 'scale_propagation' or name.endswith('.scale_propagation'):
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output: tuple(torch.zeros_like(x) for x in output)))
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
        self.emit('anchor_message_ablation', enabled=errors(p, t, valid),
                  disabled=errors(result, t, valid), change=statistics((result - p)[valid]),
                  scope='all levels Q/support disabled; direct sparse inputs and bottom prior retained')
