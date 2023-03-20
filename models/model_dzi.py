import math
import json

import numpy as np

from losses.loss import PoseLossDzi as PoseLoss
from postprocess.postprocess import PostProcessor


from models.model import PoseModule, TargetCoder

class PoseModuleDzi(PoseModule):
    def __init__(self, cfg, backbone):
        super().__init__(cfg, backbone)

        sym_types = cfg['DATASETS']['SYMMETRY_TYPES']
        diameters = cfg['DATASETS']['MESH_DIAMETERS']

       
        anchor_sizes = cfg['MODEL']['ANCHOR_SIZES']
        anchor_strides = cfg['MODEL']['ANCHOR_STRIDES']

        internal_K = cfg['INPUT']['INTERNAL_K']

        regression_type = cfg['SOLVER']['REGRESSION_TYPE']
        positive_type = cfg['SOLVER']['POSITIVE_TYPE']
        positive_num = cfg['SOLVER']['POSITIVE_NUM']
        positive_lambda = cfg['SOLVER']['POSITIVE_LAMBDA']
        loss_reg_type = cfg['SOLVER']['LOSS_REG_TYPE']
        focal_gamma = cfg['SOLVER']['FOCAL_GAMMA']
        focal_alpha = cfg['SOLVER']['FOCAL_ALPHA']
        top_k = cfg['SOLVER']['TOP_K']

        inference_th = cfg['TEST']['CONFIDENCE_TH']

        self.backbone = backbone
        
        target_coder = TargetCoder(regression_type, anchor_sizes, anchor_strides, target_type = loss_reg_type)
        
        self.post_processor = PostProcessor(inference_th, target_coder, positive_num, positive_lambda, sym_types)
        
        self.loss_evaluator = PoseLoss(
        focal_gamma, focal_alpha, anchor_sizes, anchor_strides,
        positive_type, positive_num, positive_lambda,
        top_k, internal_K, diameters, target_coder
        )
       

    def forward(self, images, targets):
        features = self.backbone(images.tensors)
        features = self.fpn(features)
        
        pred_cls, pred_reg = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            loss_list = self.loss_evaluator(pred_cls, pred_reg, targets, anchors)
            losses = {
                "loss_cls": loss_list[0],
                "loss_reg": loss_list[1],
            }
            return None, losses
        else:
            pred = self.post_processor(pred_cls, pred_reg, targets, anchors)
            return pred, {}

