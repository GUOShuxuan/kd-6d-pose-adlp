

import os
import sys
import time
import json

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import random
from arguments.argument_kd import get_args

from models.model_kd import PoseModuleKD as PoseModule


from libs.distributed import (
    get_rank,
    synchronize,
)
from tensorboardX import SummaryWriter

from libs.eval_libs import valid
from libs.train_libs import close_shared_memory, build_model, build_model_teacher
from libs.train_libs import build_dataset


# close shared memory of pytorch
if True:
    close_shared_memory()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    cfg, cfg_t = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1
    cfg_t['RUNTIME']['DISTRIBUTED'] = n_gpu > 1

    if cfg['RUNTIME']['DISTRIBUTED']:
        torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    # build dataset
    train_loader, valid_loader = build_dataset(cfg)

    cfg['KD']['vis_dir'] = cfg['RUNTIME']['WORKING_DIR']

    print('Building teacher ......')
    model_t= build_model_teacher(cfg_t, PoseModule, device)

    print('Building student ......')
    model, optimizer, scheduler, total_steps = build_model(cfg, PoseModule, device)

    VAL_FREQ = cfg['SOLVER']['VAL_FREQ']

    # 
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    # compute model size
    total_params_count = sum(p.numel() for p in model.parameters())
    total_params_count_t = sum(p.numel() for p in model_t.parameters())
    print(f"Model size: Student VS Teacher: {total_params_count:d}  vs {total_params_count_t:d}")


    # write cfg to working_dir
    with open(cfg['RUNTIME']['WORKING_DIR'] + 'cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)
        
    print("--- evaluate teacher ---")
    valid(cfg, total_steps, valid_loader, model_t, device, logger=None)
    
    model.train()
    model_t.eval()
    cfg_kd = cfg['KD']
    should_keep_training = True
    while should_keep_training:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        for idx, (images, targets, _) in pbar:
            if total_steps >= cfg['SOLVER']['MAX_ITER']:
                should_keep_training = False
                valid(cfg, total_steps, valid_loader, model, device, logger=logger)
                torch.save(model.state_dict(), cfg['RUNTIME']['WORKING_DIR'] + 'final.pth')
                print('Training finished')
                break
            total_steps += 1
            model.zero_grad()

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            with torch.no_grad(): 
                pred_t = model_t(images, targets=targets, is_teacher=True, cfg_kd=cfg_kd)

            _, loss_dict = model(images, targets=targets, pred_t=pred_t, cfg_kd=cfg_kd)

            # add pure loss value to tensorboard
            if get_rank() == 0:
                current_lr = optimizer.param_groups[0]['lr']

                # writing log to tensorboard
                if logger and idx % 10 == 0:
                    logger.add_scalar('training/learning_rate', current_lr, total_steps)
                    logger.add_scalar('training/loss_cls', loss_dict['loss_cls'], total_steps)
                    logger.add_scalar('training/loss_reg', loss_dict['loss_reg'], total_steps)
                    logger.add_scalar('training/loss_cls_reg', (loss_dict['loss_cls'] + loss_dict['loss_reg']), total_steps)
                    logger.add_scalar('training/loss_kd', loss_dict['loss_kd'], total_steps)


            loss_dict['loss_cls'] = loss_dict['loss_cls'] * cfg['SOLVER']['LOSS_WEIGHT_CLS']
            loss_dict['loss_reg'] = loss_dict['loss_reg'] * cfg['SOLVER']['LOSS_WEIGHT_REG']
            loss_cls = loss_dict['loss_cls'].mean()
            loss_reg = loss_dict['loss_reg'].mean()

            loss = loss_cls + loss_reg
            if "loss_kd" in loss_dict.keys():
                loss_dict['loss_kd']= loss_dict['loss_kd'] * cfg['KD']['LOSS_WEIGHT_KD']
                loss_kd = loss_dict['loss_kd'].mean()
                if cfg['KD']['LOSS_WEIGHT_KD'] > 0.0:
                    loss = loss + loss_kd

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['SOLVER']['GRAD_CLIP'])
            optimizer.step()
            scheduler.step()
      
            if get_rank() == 0:
                if "loss_kd" in loss_dict.keys():
                    pbar_str = (("steps: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f, kd:%.4f") % (total_steps, cfg['SOLVER']['MAX_ITER'], current_lr, loss_cls, loss_reg, loss_kd))  
                else:
                    pbar_str = (("steps: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f") % (total_steps, cfg['SOLVER']['MAX_ITER'], current_lr, loss_cls, loss_reg))
                pbar.set_description(pbar_str)
                
                if total_steps % VAL_FREQ == 0:
                    valid(cfg, total_steps, valid_loader, model, device, logger=logger)
                    model.train()
                    
                    torch.save({
                        'steps': total_steps,
                        'model': model.state_dict(), 
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        },
                        cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth',
                    )

    # output final info
    if get_rank() == 0:
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        commandstr = ' '.join([str(elem) for elem in sys.argv]) 
        final_msg = ("finished at: %s\nworking_dir: %s\ncommands:%s" % (timestr, cfg['RUNTIME']['WORKING_DIR'], commandstr))
        with open(cfg['RUNTIME']['WORKING_DIR'] + 'info.txt', 'w') as f:
            f.write(final_msg)
        print(final_msg)

    torch.cuda.empty_cache()



