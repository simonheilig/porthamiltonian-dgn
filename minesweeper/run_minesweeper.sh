#!/bin/bash

model=PHGNN_grid # specify here the model configuration name corresponding to conf.py
data=Minesweeper 
export CUDA_VISIBLE_DEVICES=0
nohup python3 -u main.py --epochs 1000 --metric roc_auc --data_name $data --model_name $model --save_dir saving_dir >out_$model\_$data 2>err_$model\_$data &
echo $! > save_pid_$data\_$model.txt

