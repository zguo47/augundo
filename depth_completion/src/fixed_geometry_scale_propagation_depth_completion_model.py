import torch

from fixed_geometry_scale_propagation_model import FixedGeometryScalePropagationDepthModel
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel


class FixedGeometryScalePropagationDepthCompletionModel(ScalePropagationDepthCompletionModel):
    '''Geometry-aware scale adaptation using the original wrapper conventions.

    Inherit the sixteen-map output layout, five-level SI/log L1 supervision,
    rejection losses, TensorBoard logging, and checkpoint handling.
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
                 detach_scaleless_depth=True,
                 use_outlier_rejection=False,
                 n_scale_refinement=1,
                 use_sparse_depth_features=False):
        del dataset_name, network_modules
        if geometry_restore_path:
            raise ValueError('This replacement has no separate geometry checkpoint; '
                             'start fresh or resume its full model checkpoint.')
        self.model_depth = FixedGeometryScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            n_propagation_iterations=n_propagation_iterations,
            encoder_type=encoder_type,
            training_stage=training_stage,
            propagation_mode=propagation_mode,
            detach_scaleless_depth=detach_scaleless_depth,
            use_outlier_rejection=use_outlier_rejection,
            n_scale_refinement=n_scale_refinement,
            use_sparse_depth_features=use_sparse_depth_features)
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.training_stage = training_stage
        self.device = device
        self.to(device)
