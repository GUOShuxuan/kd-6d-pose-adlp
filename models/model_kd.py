import math
import json

import numpy as np

import torch

from postprocess.postprocess_kd import PostProcessorKD 
from postprocess.postprocess import PostProcessor
from losses.kd_loss import KDPoseLoss
from models.model_dzi import PoseModuleDzi, TargetCoder


class PoseModuleKD(PoseModuleDzi):
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

        target_coder = TargetCoder(regression_type, anchor_sizes, anchor_strides, target_type = loss_reg_type)
    
        self.post_processor = PostProcessor(inference_th, target_coder, positive_num, positive_lambda, sym_types)

        self.post_processor_t = PostProcessorKD(inference_th, target_coder, positive_num, positive_lambda, sym_types)

        if "LEVEL" in cfg["KD"].keys():
            if cfg["KD"]["LEVEL"] == 'pred':
                self.loss_evaluator = KDPoseLoss(
                    focal_gamma, focal_alpha, anchor_sizes, anchor_strides,
                    positive_type, positive_num, positive_lambda,
                    top_k, internal_K, diameters, target_coder,
                    cfg['KD']
                    )
            else:
                raise KeyError(f'Ooops, KD from {cfg["KD"]["LEVEL"]} is not defined.')

      

    def forward(self, 
                images, 
                targets,
                is_teacher=False,
                pred_t=None, 
                cfg_kd=None):
        features = self.backbone(images.tensors)
        features = self.fpn(features)

        pred_cls, pred_reg = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
           
            loss_list = self.loss_evaluator(pred_cls, 
                                                            pred_reg, 
                                                            targets, 
                                                            anchors, 
                                                            pred_t
                                                            )
            losses = {
                "loss_cls": loss_list[0],
                "loss_reg": loss_list[1],
                "loss_kd": loss_list[2]
            }

            return None, losses
        else:
            if is_teacher:
                pred_t = dict()
                # get teacher's knowledge
                if cfg_kd['LEVEL'] == 'pred':
                    pred = self.post_processor_t(pred_cls, pred_reg, targets, anchors)
                    pred_t['post_kp_2d'] = torch.cat(pred[3], dim=0) # npos * 8*2
                    pred_t['post_kp_cls'] = torch.cat(pred[0], dim=0) # npos * 8
                    pred_t['post_pos_per_img'] = [len(pred_i) for pred_i in pred[0]]
                
                return pred_t

            pred = self.post_processor(pred_cls, pred_reg, targets, anchors)
            return pred, {}

