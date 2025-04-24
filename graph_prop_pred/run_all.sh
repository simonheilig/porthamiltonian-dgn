#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

task=ecc
data=GraphProp
save_dir=save_dir_GraphProp
model=PHGNN_GraphProp # specify here the model configuration name corresponding to conf.py

nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir $save_dir  >out_$model\_$data\_$task 2>err_$model\_$data\_$task &

#task=dist
#nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir $save_dir >out_$model\_$data\_$task 2>err_$model\_$data\_$task &

#task=diam
#nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir $save_dir >out_$model\_$data\_$task 2>err_$model\_$data\_$task &
