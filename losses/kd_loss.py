import os
import torch
from torch import nn

from losses.loss import reduce_sum, cat_boxlist, concat_box_prediction_layers
from geomloss import SamplesLoss
from losses.loss_libs import kd_loss_2d
from tools.visualizer import vis_pxpy_post_train,vis_pxpy_post_train_weight
from losses.loss import PoseLossDzi

INF = 100000000

class KDPoseLoss(PoseLossDzi):
    def __init__(self, gamma, alpha, 
                 anchor_sizes, anchor_strides, 
                 positive_type, positive_num, positive_lambda,
                 top_k, internal_K, diameters, target_coder,
                 cfg_kd=None):
        super().__init__(gamma, alpha, 
                    anchor_sizes, anchor_strides, 
                    positive_type, positive_num, positive_lambda,
                    top_k, internal_K, diameters, target_coder)
        
        if cfg_kd is not None:
            self.cfg_kd = cfg_kd
            self.kd_loss = SamplesLoss(self.cfg_kd['GTYPE'], 
                                         p=self.cfg_kd['GP'], 
                                         blur=self.cfg_kd['GBLUR'], 
                                         scaling=self.cfg_kd['SCALING'], 
                                         reach=self.cfg_kd['REACH'])
            
            self.weighted_ot = self.cfg_kd['WEIGHTED_OT']
            self.wot_detach = self.cfg_kd['DETACH']
            if 'vis_dir' in self.cfg_kd.keys():
                self.step = 0
                self.vis_dir = self.cfg_kd['vis_dir'] + '/vis'
                os.makedirs(self.vis_dir, exist_ok=True)


    def KDObjectSpaceLoss(self, pred, target_2D, target_3D_in_camera_frame, cls_labels, anchors, pred_t, bbox_trans, weight=None):
        if not isinstance(self.diameters, torch.Tensor):
            self.diameters = torch.FloatTensor(self.diameters).to(device=pred.device).view(-1)
        self.cls_id = torch.unique(cls_labels)
        diameter_ext = self.diameters[cls_labels.view(-1,1).repeat(1, 8*3).view(-1, 3, 1)]

        cellNum = pred.shape[0]
        pred_filtered = pred.view(cellNum, -1, 16)[torch.arange(cellNum), cls_labels]

        pred_xy = self.target_coder.decode(pred_filtered, anchors, bbox_trans)
        pred_xy = pred_xy.view(-1,2,8).transpose(1,2).contiguous().view(-1,2)

        ## 2d 
        target_xy = self.target_coder.decode(target_2D, anchors, bbox_trans)
        target_xy = target_xy.view(-1,2,8).transpose(1,2).contiguous().view(-1,2)

        # construct normalized 2d
        B = torch.inverse(self.internal_K).mm(torch.cat((pred_xy.t(), torch.ones_like(pred_xy[:,0]).view(1,-1)), dim=0)).t()
        # compute projection matrices
        P = torch.bmm(B.view(-1, 3, 1), B.view(-1, 1, 3)) / torch.bmm(B.view(-1, 1, 3), B.view(-1, 3, 1))

        target_3D_in_camera_frame_ = target_3D_in_camera_frame
        target_3D_in_camera_frame = target_3D_in_camera_frame.view(-1, 3, 1)
        px = torch.bmm(P, target_3D_in_camera_frame)

        
        target_3D_in_camera_frame = target_3D_in_camera_frame / diameter_ext
        px = px / diameter_ext
    
        scaling_factor = 50 # 0.02d
        losses = nn.SmoothL1Loss(reduction='none')(scaling_factor * px, scaling_factor * target_3D_in_camera_frame).view(cellNum, -1).mean(dim=1)
        losses = losses / scaling_factor

        ## --- loss_kd
        pos_per_img_t = pred_t['post_pos_per_img']
        pos_per_img_ = self.pos_per_img
        pos_per_img_t_ = pos_per_img_t

        if self.cfg_kd['GnD'] == 2:
            pred_t_xy = pred_t['post_kp_2d'].view(-1, 2)

            if self.weighted_ot:
                t_cls = pred_t['post_kp_cls'].pow(2)
                s_cls = torch.broadcast_to(self.pred_cls[..., self.cls_id], (self.pred_cls.size(0), 8))
                if self.wot_detach: #True: with detach
                    s_cls = s_cls.detach() # s_cls.requires_grad 
                loss_kd = kd_loss_2d(pred_xy, pred_t_xy, s_cls, t_cls, self.w, self.h, self.cfg_kd['GLEVEL'], self.kd_loss, dim=2, pos_per_img=pos_per_img_, pos_per_img_t=pos_per_img_t_)

                if self.step ==0 or (self.step + 1) % 1000 == 0:
                    vis_pxpy_post_train_weight(pred_xy, pred_t_xy, s_cls.reshape(-1, 1), t_cls.reshape(-1, 1), self.step, save_dir=self.vis_dir, pos_per_img_1=pos_per_img_, pos_per_img_2=pos_per_img_t_, loss=loss_kd)

            else:
                t_cls = None
                s_cls = None
                loss_kd = kd_loss_2d(pred_xy, pred_t_xy, s_cls, t_cls, self.w, self.h, self.cfg_kd['GLEVEL'], self.kd_loss, dim=2, pos_per_img=pos_per_img_, pos_per_img_t=pos_per_img_t_)
                
                if self.step ==0 or (self.step + 1) % 1000 == 0:
                    vis_pxpy_post_train(pred_xy, pred_t_xy, self.step, save_dir=self.vis_dir, pos_per_img_1=pos_per_img_, pos_per_img_2=pos_per_img_t_, loss=loss_kd)

        if len(loss_kd) > 0:
            loss_kd = sum(loss_kd)/ len(loss_kd)
            loss_kd = loss_kd.sum()
        else:
            loss_kd = torch.tensor(0.).cuda()

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum(), loss_kd

    def __call__(self, pred_cls, pred_reg, targets, anchors, pred_t):
        labels, reg_targets, aux_raw_boxes, aux_3D_in_camera_frame, aux_bbox_trans = self.prepare_targets(targets, anchors)

        N = len(labels)
        self.batch_size = N
        self.h = 480 # targets[0].height not 256
        self.w = 640 # targets[0].width 

        pred_cls_flatten, pred_reg_flatten = concat_box_prediction_layers(pred_cls, pred_reg)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        aux_raw_boxes_flatten = torch.cat(aux_raw_boxes, dim=0)
        aux_3D_in_camera_frame_flatten = torch.cat(aux_3D_in_camera_frame, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)
        aux_bbox_trans_flatten = torch.cat(aux_bbox_trans, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        
        # num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        
        valid_cls_inds = torch.nonzero(labels_flatten >= 0).squeeze(1)
        cls_loss = self.cls_loss_func(pred_cls_flatten[valid_cls_inds], labels_flatten[valid_cls_inds]) 

        if pos_inds.numel() > 0:
            pred_reg_flatten = pred_reg_flatten[pos_inds]
            cls_label_flatten = labels_flatten[pos_inds] - 1 # start from class 0
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            aux_raw_boxes_flatten = aux_raw_boxes_flatten[pos_inds]
            aux_3D_in_camera_frame_flatten = aux_3D_in_camera_frame_flatten[pos_inds]
            anchors_flatten = anchors_flatten[pos_inds]
            aux_bbox_trans_flatten = aux_bbox_trans_flatten[pos_inds]

            self.pos_per_img = [(label_> 0).sum().item() for label_ in labels]
            assert sum(self.pos_per_img) == total_num_pos

            if self.target_coder.target_type == '2D':
                reg_loss = self.ImageSpaceLoss(pred_reg_flatten, reg_targets_flatten, cls_label_flatten, anchors_flatten)
            elif self.target_coder.target_type == '3D':
                if self.weighted_ot:
                    self.pred_cls = torch.clamp(torch.sigmoid(pred_cls_flatten[pos_inds]), min=10e-4, max=1-10e-4)
                reg_loss, kd_loss = self.KDObjectSpaceLoss(pred_reg_flatten, reg_targets_flatten, aux_3D_in_camera_frame_flatten, cls_label_flatten, anchors_flatten, pred_t, aux_bbox_trans_flatten) 
            else:
                assert(0)
        else:
            reg_loss = pred_reg_flatten.sum()
            kd_loss = pred_reg_flatten.sum()

        self.step += 1
        return [cls_loss, reg_loss, kd_loss] 