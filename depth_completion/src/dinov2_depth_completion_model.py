import warnings

import torch
import torch.nn as nn

from dinov2_feedforward_model import DINOv2DepthCompletionModel
from utils.src import loss_utils


EPSILON = 1e-8


class DINOv2GuidedModel(object):
    """DinoV2-guided depth completion model."""

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        network_modules = [] if network_modules is None else network_modules

        dinov2_model_name = 'dinov2_vits14'
        dinov2_repository = 'facebookresearch/dinov2:f896929'

        self.model_depth = DINOv2DepthCompletionModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            dinov2_model_name=dinov2_model_name,
            dinov2_repository=dinov2_repository,
            freeze_dinov2=True)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device
        self.to(device)

    def forward_depth(self,
                      image,
                      sparse_depth,
                      validity_map,
                      intrinsics=None,
                      return_all_outputs=False):
        '''
        Forwards depth through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            return_all_outputs : bool
                if set, then return list of all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        output_depth = self.model_depth.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map=validity_map)

        return [output_depth] if return_all_outputs else output_depth

    def compute_loss_supervised(self, 
                                target_depth, 
                                output_depth, 
                                w_losses):
        '''
        Computes supervised loss
        '''
        output_depth = output_depth[0]
        validity = torch.logical_and(
            torch.isfinite(target_depth),
            target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            torch.clamp(
                target_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        loss_log_l1 = loss_utils.log_l1_loss_func(
            src=output_depth,
            tgt=target_depth,
            w=validity)
        w_supervised = w_losses.get('w_supervised', 1.0)
        loss = w_supervised * loss_log_l1

        return loss, {
            'loss': loss,
            'loss_log_l1': loss_log_l1
        }

    def parameters(self):
        return list(self.model_depth.parameters())

    def parameters_depth(self):
        return self.parameters()

    def train(self):
        self.model_depth.train()

    def eval(self):
        self.model_depth.eval()

    def to(self, device):
        self.device = device
        self.model_depth.to(device)

    def data_parallel(self):
        if not isinstance(self.model_depth, nn.DataParallel):
            self.model_depth = nn.DataParallel(self.model_depth)

    def _model_state_dict(self):
        if isinstance(self.model_depth, nn.DataParallel):
            return self.model_depth.module.state_dict()
        return self.model_depth.state_dict()

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Saves model checkpoint
        '''
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

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (ValueError, RuntimeError) as error:
                warnings.warn(
                    'Unable to restore optimizer state: {}'.format(error))

        return checkpoint.get('train_step', 0), optimizer
