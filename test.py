import os
import sys
import time
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from arguments.argument import get_args
from backbone import darknet_tiny, darknet53, darknet_tiny_h
from libs.dataset import BOP_Dataset, collate_fn
from models.model import PoseModule
import libs.transform as transform
from libs.distributed import (
    get_rank,
    synchronize
)
from libs.train_libs import data_sampler
from libs.eval_libs import valid


from tensorboardX import SummaryWriter

# reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':

    cfg = get_args()

    # create working_dir dynamically
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
    working_dir = 'working_dirs' + '/' + name_wo_ext + '/' + timestr + '/'
    cfg['RUNTIME']['WORKING_DIR'] = working_dir
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1

    if cfg['RUNTIME']['DISTRIBUTED']:
        torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

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

    valid_set = BOP_Dataset(
        cfg['DATASETS']['TEST'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False,
        DZI=True)

    
    if cfg['MODEL']['BACKBONE'] == 'darknet_tiny':
        backbone = darknet_tiny(pretrained=False)
    elif cfg['MODEL']['BACKBONE'] == 'darknet_tiny_h':
        backbone = darknet_tiny_h(pretrained=False)
    elif cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=False)
    model = PoseModule(cfg, backbone)

    # load weight
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                chkpt = chkpt['model']
            # model.load_state_dict(chkpt) # strict
            # loose loading
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in chkpt.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            # 
            print('Weights are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            print('Loading weights from %s is failed' % (cfg['RUNTIME']['WEIGHT_FILE']))
            print("Random initialized weights.")
    else:
        print("Random initialized weights.")

    model = model.to(device)
    
    batch_size_per_gpu = int(cfg['TEST']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])
    batch_size_per_gpu = 24
    if batch_size_per_gpu == 0:
        print('ERROR: %d GPUs for %d batch(es)' % (cfg['RUNTIME']['N_GPU'], cfg['TEST']['IMS_PER_BATCH']))
        assert(0)

    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module
 
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range = \
        valid(cfg, 0, valid_loader, model, device, logger=logger)




