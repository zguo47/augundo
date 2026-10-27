import torch
import torch.nn as nn

from fixed_geometry_scale_propagation_model import FixedGeometryScalePropagationDepthModel
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from utils.src import loss_utils


class FixedGeometryScalePropagationDepthCompletionModel(ScalePropagationDepthCompletionModel):
    '''Repository wrapper for the independent fixed-geometry baseline.

    Reuse only the repository-facing map layout, device methods, and common
    call signatures. The network, stage-specific losses, and checkpoint
    initialization below are separate from the current scale-propagation model.
    '''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda'),
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 encoder_type='conv',
                 training_stage='joint',
                 propagation_mode='hierarchical',
                 geometry_restore_path=None,
                 detach_scaleless_depth=True):
        del dataset_name, network_modules
        self.model_depth = FixedGeometryScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            n_propagation_iterations=n_propagation_iterations,
            encoder_type=encoder_type,
            training_stage=training_stage,
            propagation_mode=propagation_mode,
            detach_scaleless_depth=detach_scaleless_depth)
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.training_stage = training_stage
        self.geometry_restore_path = geometry_restore_path
        self.device = device
        self.to(device)
        if geometry_restore_path:
            # Start a NEW scale experiment: load geometry only, not the first
            # stage's optimizer, training step, or unused propagation weights.
            checkpoint = torch.load(geometry_restore_path, map_location=device)
            self.model_depth.geometry.load_state_dict(checkpoint['geometry_state_dict'])

    def compute_loss_supervised(self, target_depth, output_depth, w_losses):
        '''Train SI geometry and metric scale together, unless a stage is selected.'''
        validity = (target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            target_depth.clamp(self.min_predict_depth, self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        # There is ONE predicted geometry, so supervise its full-resolution
        # output once. Coarser copies are not additional learned depth heads.
        loss_si = loss_utils.scale_invariant_loss_func(
            src=output_depth[1], tgt=target_depth, w=validity)
        loss_log_l1 = target_depth.new_zeros(())
        loss_info = {'loss_si_r0': loss_si}
        for resolution, metric in enumerate(output_depth[11:16]):
            value = loss_utils.log_l1_loss_func(src=metric, tgt=target_depth, w=validity)
            loss_log_l1 = loss_log_l1 + value / (4 ** resolution)
            loss_info['loss_log_l1_r{}'.format(resolution)] = value

        w_supervised = w_losses.get('w_supervised', 1.0)
        # Joint mode combines both objectives in the same optimizer step.
        # Separate-stage ablations still report, but do not train, the other.
        loss_si_weighted = w_supervised * w_losses.get('w_si', 1.0) * loss_si \
            if self.training_stage in ['joint', 'geometry'] else loss_si.detach() * 0.0
        loss_log_l1_weighted = w_supervised * w_losses.get('w_log_l1', 1.0) * loss_log_l1 \
            if self.training_stage in ['joint', 'scale'] else loss_log_l1.detach() * 0.0
        loss = loss_si_weighted + loss_log_l1_weighted
        loss_info.update(
            loss=loss, loss_si=loss_si, loss_log_l1=loss_log_l1,
            loss_si_weighted=loss_si_weighted,
            loss_log_l1_weighted=loss_log_l1_weighted)
        return loss, loss_info

    def parameters(self):
        # Adam receives only the stage's trainable parameters. In particular,
        # frozen geometry cannot change through weight decay or optimizer state.
        return [parameter for parameter in self.model_depth.parameters()
                if parameter.requires_grad]

    def save_model(self, checkpoint_path, step, optimizer=None):
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) else self.model_depth
        checkpoint = dict(
            train_step=step,
            model_state_dict=model.state_dict(),
            geometry_state_dict=model.geometry.state_dict(),
            fixed_geometry_stage=self.training_stage)
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
