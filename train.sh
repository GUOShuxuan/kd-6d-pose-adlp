# darknet_tiny_h
## Student-baseline: kd_weight = 0.0
python3 train_kd.py --config_file ./configs/ape.yaml --config_file_t ./configs/ape.yaml --backbone darknet_tiny_h --backbone_t darknet53 --weight_file_t model_t_ape/final.pth --kd_weight 0. --max_iters 10000 --working_dir outputs/ape/baseline 2>&1 | tee outputs/ape/baseline/exp.log 

## our kd: kd_weight = 5.0
python3 train_kd.py --config_file ./configs/ape.yaml --config_file_t ./configs/ape.yaml --backbone darknet_tiny_h --backbone_t darknet53 --weight_file_t model_t_ape/final.pth --kd_weight 5. --max_iters 10000 --working_dir outputs/ape/kd 2>&1 | tee outputs/ape/kd/exp.log

# Teacher
python3 train_kd.py --config_file ./configs/ape.yaml --config_file_t ./configs/ape.yaml --backbone darknet53 --backbone_t darknet53 --weight_file_t None --kd_weight 0.  --working_dir outputs/ape/darknet53 2>&1 | tee outputs/ape/darknet53/exp.log 
