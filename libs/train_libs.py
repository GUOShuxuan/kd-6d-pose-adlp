from logging import raiseExceptions
import os
import sys
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
import numpy as np
import random


from backbone import darknet_tiny, darknet53, darknet_tiny_h
from libs.dataset import BOP_Dataset, collate_fn

import libs.transform as transform

from libs.distributed import (
    get_rank,
    DistributedSampler,
    all_gather,
)

from tensorboardX import SummaryWriter


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def close_shared_memory():
    from torch.utils.data import dataloader
    from torch.multiprocessing import reductions
    from multiprocessing.reduction import ForkingPickler
    default_collate_func = dataloader.default_collate
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
    setattr(dataloader, 'default_collate', default_collate_override)
    for t in torch._storage_classes:
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]


def accumulate_dicts(data):
    all_data = all_gather(data)

    if get_rank() != 0:
        return

    data = {}

    for d in all_data:
        data.update(d)

    return data

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def build_backbone(arch):
    
    if arch == 'darknet_tiny':
        backbone = darknet_tiny(pretrained=True)
    elif arch == 'darknet_tiny_h':
        backbone = darknet_tiny_h(pretrained=False) # don't have the pretrained weights
    elif arch == 'darknet53':
        backbone = darknet53(pretrained=True)

    return backbone

def build_model(cfg, posemodule, device='cuda'):
  
    backbone = build_backbone(cfg['MODEL']['BACKBONE'])
    model = posemodule(cfg, backbone)

    # load weight from config file
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                chkpt = chkpt['model']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in chkpt.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            print('Weights are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            print('Loading weights from %s is failed' % (cfg['RUNTIME']['WEIGHT_FILE']))
            print("Random initialized weights.")
    else:
        print("-- Random initialized weights.")

    model = model.to(device)

 
    
    base_lr = cfg['SOLVER']['BASE_LR'] / cfg['RUNTIME']['N_GPU']
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, base_lr, cfg['SOLVER']['MAX_ITER']+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module

   
    config_str = (cfg['RUNTIME']['CONFIG_FILE']).split('/') #['.', 'configs', 'linemod', 'ape.yaml']
    dataset_str = config_str[2]
    # class_str = os.path.splitext(config_str[3])[0]

    if len(cfg['RUNTIME']['WORKING_DIR']) == 0:
        # create working_dir dynamically
        class_str = os.path.splitext(config_str[3])[0]
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        #--config_file ./configs/linemod/ape.yaml
        name_wo_ext = f"{dataset_str}/{cfg['MODEL']['BACKBONE']}/{class_str}"
        cfg['RUNTIME']['WORKING_DIR'] = f"outputs/{name_wo_ext}/{timestr}/"
    # resume from a checkpoint in the working_dir, model+optimizer+scheduler
    preload_file_name = None
    if os.path.exists(cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth'):
        preload_file_name = cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth'
    elif os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        preload_file_name = cfg['RUNTIME']['WEIGHT_FILE']
    total_steps = 0.
    if preload_file_name:
        try:
            chkpt = torch.load(preload_file_name, map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                assert('steps' in chkpt and 'optim' in chkpt)
                total_steps = chkpt['steps']
                model.load_state_dict(chkpt['model'])
                optimizer.load_state_dict(chkpt['optim'])
                scheduler.load_state_dict(chkpt['sched'])
                print('Weights, optimzer, scheduler are loaded from %s, starting from step %d' % (preload_file_name, total_steps))
                
            else:
                model.load_state_dict(chkpt)
                print('Weights from are loaded from ' + preload_file_name)
        except:
            pass
    else:
        pass

    return model, optimizer, scheduler, total_steps

def build_model_teacher(cfg, posemodule, device):
    backbone = build_backbone(cfg['MODEL']['BACKBONE'])
    model = posemodule(cfg, backbone)

    # load weight from config file
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                chkpt = chkpt['model']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in chkpt.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            print('Weights are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            print('Loading weights from %s is failed' % (cfg['RUNTIME']['WEIGHT_FILE']))
            print("Random initialized weights.")
    else:
        print("Random initialized weights.")


    model = model.to(device)

    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module
    

    return model


def build_dataset(cfg):
    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

    train_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            transform.RandomOcclusion(cfg['SOLVER']['AUGMENTATION_OCCLUSION']),
            transform.RandomBackground(cfg['SOLVER']['AUGMENTATION_BACKGROUND_DIR']),
            transform.RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'], 
                cfg['SOLVER']['AUGMENTATION_SCALE'], 
                cfg['SOLVER']['AUGMENTATION_ROTATION'], 
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.RandomHSV(
                cfg['SOLVER']['AUGMENTATION_ColorH'], 
                cfg['SOLVER']['AUGMENTATION_ColorS'], 
                cfg['SOLVER']['AUGMENTATION_ColorV']
                ),
            transform.RandomPencilSharpen(cfg['SOLVER']['AUGMENTATION_Sharpen']),
            transform.RandomSmooth(cfg['SOLVER']['AUGMENTATION_Smooth']),
            transform.RandomNoise(cfg['SOLVER']['AUGMENTATION_Noise']),
            transform.Grayscalize(cfg['SOLVER']['AUGMENTATION_Grayscalize']),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(),
        ]
    )

    valid_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.Grayscalize(cfg['SOLVER']['AUGMENTATION_Grayscalize']),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(), 
        ]
    )

    train_set = BOP_Dataset(
        cfg['DATASETS']['TRAIN'], 
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        train_trans,
        cfg['DATASETS']['SYMMETRY_TYPES'],
        training = True,
        DZI=True)
    valid_set = BOP_Dataset(
        cfg['DATASETS']['VALID'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False,
        DZI=True)
    
    batch_size_per_gpu = int(cfg['SOLVER']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(train_set, shuffle=True, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        worker_init_fn=seed_worker,
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        worker_init_fn=seed_worker,
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    
    return train_loader, valid_loader







