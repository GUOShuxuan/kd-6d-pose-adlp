import argparse
import yaml
from arguments.argument import custom_cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--config_file', type=str, default='./configs/ape.yaml')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--working_dir', type=str, default='./outputs/')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--weight_file', type=str, default='')
    parser.add_argument('--running_device', type=str, default='cuda')

    # add for baselines
    parser.add_argument('--backbone', type=str, default='darknet_tiny_h')
    parser.add_argument('--max_iters',  type=int, default=20000, help='max iteration')
    parser.add_argument('--base_lr',  type=float, default=0.001, help='base learning rate')

     # add teacher config 
    parser.add_argument('--config_file_t', type=str, default='./configs/occ_linemod.yaml')
    parser.add_argument('--backbone_t', type=str, default='darknet53')
    parser.add_argument('--weight_file_t', type=str, default='')
    
    # for kd
    parser.add_argument('--kd_weight', type=float, default=5, help='weight of loss_kd_loss')
    parser.add_argument('--kd_level', type=str, default='pred', help='level to be distilled')

    # for loss_kd_loss
    parser.add_argument('--gtype',  type=str, default='sinkhorn', help='function of kd loss', choices=['l1', 'l2', 'sinkhorn', 'gaussian', 'laplacian', 'energy'])
    parser.add_argument('--glevel',  type=str, default='point', help='level of kd loss', choices=['point'])
    parser.add_argument('--p',  type=float, default=2.0, help='p of loss_kd_loss')
    parser.add_argument('--blur',  type=float, default=0.001, help='blur of loss_kd_loss')
    parser.add_argument('--gnD',  type=int, default=2, help='dimensions of loss_kd_loss')
    parser.add_argument('--weightedOT',  type=str2bool, nargs='?', const=True, default=True, help='weighted OT of loss_kd_loss')
    parser.add_argument('--wot_detach', type=str2bool, nargs='?', const=True, default=False, help='weighted ot with detached cls')
    parser.add_argument('--scaling', type=float, default=0.5, help='param for sinkhorn loss')
    parser.add_argument('--reach', type=float, default=0.5, help='param for sinkhorn loss')
    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()

    # Read yaml configure
    with open(args.config_file, 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)

    cfg['RUNTIME'] = {}
    cfg['RUNTIME']['LOCAL_RANK'] = args.local_rank
    cfg['RUNTIME']['CONFIG_FILE'] = args.config_file
    cfg['RUNTIME']['NUM_WORKERS'] = args.num_workers
    cfg['RUNTIME']['WEIGHT_FILE'] = args.weight_file
    cfg['RUNTIME']['WORKING_DIR'] = args.working_dir
    cfg['RUNTIME']['RUNNING_DEVICE'] = args.running_device
    #
    if len(args.test_file) > 0:
        cfg['DATASETS']['TEST'] = args.test_file

    cfg['MODEL']['BACKBONE'] = args.backbone
    cfg = custom_cfg(cfg)
    cfg['SOLVER']['MAX_ITER'] = args.max_iters
    cfg['SOLVER']['BASE_LR'] = args.base_lr

    # kd config setting
    cfg['KD']['LOSS_WEIGHT_KD']=args.kd_weight
    cfg['KD']['LEVEL'] = args.kd_level

    if cfg['KD']['LEVEL'] == 'pred':
        cfg['KD']['GLEVEL'] = args.glevel
        cfg['KD']['GTYPE'] = args.gtype
        cfg['KD']['GP'] = args.p
        cfg['KD']['GBLUR'] = args.blur
        cfg['KD']['GnD'] = args.gnD
        cfg['KD']['WEIGHTED_OT'] = args.weightedOT
        cfg['KD']['DETACH'] = args.wot_detach
        cfg['KD']['SCALING'] = args.scaling
        cfg['KD']['REACH'] = args.reach
    
    ## add teacher config
    with open(args.config_file_t, 'r') as load_f:
        cfg_t = yaml.load(load_f, Loader=yaml.FullLoader)

    cfg_t['RUNTIME'] = {}
    cfg_t['RUNTIME']['LOCAL_RANK'] = args.local_rank
    cfg_t['RUNTIME']['CONFIG_FILE'] = args.config_file_t
    cfg_t['RUNTIME']['NUM_WORKERS'] = args.num_workers
    cfg_t['RUNTIME']['WEIGHT_FILE'] = args.weight_file_t
    cfg_t['RUNTIME']['RUNNING_DEVICE'] = args.running_device

    # set backbone
    cfg_t['MODEL']['BACKBONE'] = args.backbone_t
    cfg_t = custom_cfg(cfg_t)

    return cfg, cfg_t