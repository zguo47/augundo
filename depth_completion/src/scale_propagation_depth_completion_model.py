import torch
import torch.nn as nn
import torch.nn.functional as functional

from scale_propagation_model import ScalePropagationDepthModel
from utils.src import loss_utils


class ScalePropagationDepthCompletionModel(object):
    '''Repository wrapper for shared-encoder multi-scale propagation.'''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):
        del dataset_name, network_modules

        self.model_depth = ScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device
        self.to(device)

    def forward_depth(self,
                      image,
                      sparse_depth=None,
                      validity_map=None,
                      intrinsics=None,
                      return_all_outputs=False):
        del validity_map, intrinsics

        output_depths = self.model_depth(
            image=image,
            sparse_depth=sparse_depth)
        return output_depths if return_all_outputs else output_depths[0]

    def compute_loss_supervised(self, target_depth, output_depth, w_losses):
        '''Supervise the final depth and valid bottom propagation values.'''
        final_depth = output_depth[0]
        coarse_depth = output_depth[1]

        validity = (target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            torch.clamp(
                target_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        loss_final = loss_utils.log_l1_loss_func(
            src=final_depth,
            tgt=target_depth,
            w=validity)

        coarse_validity = functional.adaptive_avg_pool2d(
            validity,
            output_size=coarse_depth.shape[-2:])
        coarse_target = functional.adaptive_avg_pool2d(
            target_depth * validity,
            output_size=coarse_depth.shape[-2:]) / (coarse_validity + 1e-7)
        coarse_validity = \
            (coarse_validity > 0.0).to(target_depth.dtype) * \
            (coarse_depth > 0.0).to(target_depth.dtype)
        coarse_depth = torch.where(
            coarse_validity > 0.0,
            torch.clamp(
                coarse_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(coarse_depth, self.min_predict_depth))
        coarse_target = torch.where(
            coarse_validity > 0.0,
            torch.clamp(
                coarse_target,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(coarse_target, self.min_predict_depth))
        loss_coarse = loss_utils.log_l1_loss_func(
            src=coarse_depth,
            tgt=coarse_target,
            w=coarse_validity)

        w_supervised = w_losses.get('w_supervised', 1.0)
        w_coarse = w_losses.get('w_coarse', 1.0)
        loss = w_supervised * (loss_final + w_coarse * loss_coarse)

        return loss, {
            'loss': loss,
            'loss_log_l1': loss_final,
            'loss_coarse': loss_coarse
        }

    def parameters(self):
        return list(self.model_depth.parameters())

    def parameters_depth(self):
        return self.parameters()

    def parameters_pose(self):
        return []

    def forward_pose(self, image0, image1):
        del image0, image1
        return None

    def train(self):
        self.model_depth.train()

    def eval(self):
        self.model_depth.eval()

    def to(self, device):
        self.device = device
        self.model_depth.to(device)

    def data_parallel(self):
        self.model_depth = nn.DataParallel(self.model_depth)

    def _model_state_dict(self):
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) \
            else self.model_depth
        return model.state_dict()

    def save_model(self, checkpoint_path, step, optimizer=None):
        checkpoint = {
            'train_step': step,
            'model_state_dict': self._model_state_dict()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, restore_path, optimizer=None):
        checkpoint = torch.load(restore_path, map_location=self.device)
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) \
            else self.model_depth
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('train_step', 0), optimizer
