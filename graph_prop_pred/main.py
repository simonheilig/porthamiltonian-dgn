import os
import torch

import ray
import time
import argparse
import datetime
from utils import DATA
from conf import CONFIGS
from utils.pna_dataset import TASKS
from model_selection import model_selection
from utils.io import create_if_not_exist, join

# Ignore warnings
from sklearn.exceptions import UndefinedMetricWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

torch.autograd.set_detect_anomaly(True)

print('Settings:')
print('\tKMP_SETTING:', os.environ.get('KMP_SETTING'))
print('\tOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print('\tKMP_BLOCKTIME:', os.environ.get('KMP_BLOCKTIME'))
print('\tMALLOC_CONF:', os.environ.get('MALLOC_CONF'))
print('\tLD_PRELOAD:', os.environ.get('LD_PRELOAD'))
print()

if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  

    parser.add_argument('--data_name', 
                        help='The name of the dataset to load.',
                        default='GraphProp',
                        choices=DATA)
    parser.add_argument('--task', 
                        help='The name of the GraphProp task.',
                        default=None,
                        choices=TASKS)
    parser.add_argument('--model_name',
                        help='The model name.',
                        default='GraphAntiSymmetricNN',
                        choices=CONFIGS.keys())
    parser.add_argument('--epochs', help='The number of epochs.', default=1500, type=int)
    parser.add_argument('--early_stopping', 
                        help='Training stops if the selected metric does not improve for X epochs',
                        type=int,
                        default=100)
    parser.add_argument('--save_dir', help='The saving directory.', default='.')
    parser.add_argument('--dtype', help="Set the floating point precision",default="32",choices=["32","64"])
    parser.add_argument('--par', help="Set the degree of maximum parallelism", type=int, default=1)
    args = parser.parse_args()
    
    print(args)
    assert args.par > 0 and (args.par % 1 == 0), "The degree of parallelism should be greather than 0 and cannot be a fraction"
    assert args.data_name in args.model_name, f"the selected model doesn't match the selcted data. Got {args.data_name} and {args.model_name}"
    assert args.epochs >= 1, 'The number of epochs should be greather than 0'
    args.save_dir = os.path.abspath(args.save_dir)
    dtype = torch.float32 if args.dtype == "32" else torch.float64
    ray.init(num_cpus=args.par) # local ray initialization
    p = join(args.save_dir, args.data_name)
    create_if_not_exist(p)    
    print(args.task)
    if args.task is not None:
        p = join(p, args.task)
        create_if_not_exist(p)
    exp_dir = join(p, args.model_name)
    create_if_not_exist(exp_dir)

    
    # Run model selection
    best_conf_res = model_selection(model_name = args.model_name,
                            data_name = args.data_name,
                            early_stopping_patience = args.early_stopping,
                            epochs = args.epochs,
                            task = args.task,
                            data_dir = args.save_dir,
                            exp_dir = exp_dir,
                            dtype=dtype, 
                            parallelism = args.par)

    print(best_conf_res)
    elapsed = time.time() - t0
    print("Elapsed Time: ",datetime.timedelta(seconds=elapsed))
