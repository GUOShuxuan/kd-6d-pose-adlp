from logging import raiseExceptions
import os
import sys
import time
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

import numpy as np
import cv2
import math


import libs.transform as transform
from libs.evaluate import (
    evaluate_pose_predictions, 
    remap_predictions
)
from libs.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from libs.utils import (
    load_bop_meshes,
    load_bbox_3d,
    visualize_pred,
    pose_symmetry_handling,
    print_accuracy_per_class,
    network_grad_ratio,
    json_dumps_numpy,
)

from libs.train_libs import accumulate_dicts

@torch.no_grad()
def valid(cfg, steps, loader, model, device, logger=None):
    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    meshes, _ = load_bop_meshes(cfg['DATASETS']['MESH_DIR'])
    bboxes_3d = load_bbox_3d(cfg['DATASETS']['BBOX_FILE'])

    for bIdx, (images, targets, meta_infos) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        pred, aux = model(images, targets=targets)

        # pred = [p.to('cpu') for p in pred]
        iIdx = 0
        for m, p in zip(meta_infos, pred):
            iIdx += 1
            new_p = remap_predictions(
                cfg['INPUT']['INTERNAL_K'], 
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                bboxes_3d, m, p
                )
           

            if len(new_p) == 0:
                preds.update({m['path']:{
                    'meta': m,
                    'pred': new_p
                }})
            else:
                preds.update({m['path']:{
                    'meta': m,
                    'pred': [new_p[0][:-1]]
                }})
        #

    preds = accumulate_dicts(preds)

    if get_rank() != 0:
        return

    # write predictions to file
    json_file_name = cfg['RUNTIME']['WORKING_DIR'] + "preds.json"
    preds_formatted = json_dumps_numpy(preds)
    with open(json_file_name, 'w') as f:
        f.write(preds_formatted)
    # reload
    with open(json_file_name, 'r') as f:
        preds = json.load(f)
        
    accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate_pose_predictions(preds, cfg['DATASETS']['N_CLASS'], meshes, cfg['DATASETS']['MESH_DIAMETERS'], cfg['DATASETS']['SYMMETRY_TYPES'])

    print_accuracy_per_class(accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class)
    # print(accuracy_adi_per_depth)

    # writing log to tensorboard
    if logger:
        classNum = cfg['DATASETS']['N_CLASS'] - 1 # get rid of background class        
        assert(len(accuracy_adi_per_class) == classNum)
        assert(len(accuracy_rep_per_class) == classNum)

        all_adi = {}
        all_rep = {}
        validClassNum = 0

        for i in range(classNum):
            className = ('class_%02d' % i)
            logger.add_scalars('ADI/' + className, accuracy_adi_per_class[i], steps)
            logger.add_scalars('REP/' + className, accuracy_rep_per_class[i], steps)
            # 
            assert(len(accuracy_adi_per_class[i]) == len(accuracy_rep_per_class[i]))
            if len(accuracy_adi_per_class[i]) > 0:
                for key, val in accuracy_adi_per_class[i].items():
                    if key in all_adi:
                        all_adi[key] += val
                    else:
                        all_adi[key] = val
                for key, val in accuracy_rep_per_class[i].items():
                    if key in all_rep:
                        all_rep[key] += val
                    else:
                        all_rep[key] = val
                validClassNum += 1

        # averaging
        for key, val in all_adi.items():
            all_adi[key] = val / validClassNum
        for key, val in all_rep.items():
            all_rep[key] = val / validClassNum  
        logger.add_scalars('ADI/all_class', all_adi, steps)
        logger.add_scalars('REP/all_class', all_rep, steps)

    
    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range




