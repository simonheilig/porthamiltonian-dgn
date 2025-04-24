import torch

import os
import ray
import tqdm
import pandas as pd
from conf import CONFIGS
from utils import get_dataset
from train_GraphProp import train_val_pipeline_GraphProp
from utils.io import load, dump, join, create_if_not_exist
from typing import Optional
from utils.pna_dataset import NODE_LVL_TASKS


def wait_and_collect(ray_ids,ids_to_configs, df, exp_dir, tqdm_ids, data_name, final_json, best_score):
    print("wait")
    done_id, ray_ids = ray.wait(ray_ids)
    id_ = done_id[0]

    res = ray.get(id_)
    
    conf = ids_to_configs[id_]
    result = {} #'ray_id': id_}
    for key_name, values in conf.items():
        if isinstance(values, dict):
            for k, v in values.items():
                result[f'{key_name}_{k}'] = v
        else:
            result[key_name] = values

    avg = res['avg_res']
    result.update(avg)
    df.append(result)
    pd.DataFrame(df).sort_values('avg best_val_score').to_csv(join(exp_dir, 'partial_results.csv'), index=False)

    tqdm_ids.update(1)
    if data_name == 'GraphProp' or "toy" in data_name or "Peptides" in data_name:
        if best_score is None or avg['avg best_val_score'] < best_score:
            best_score = avg['avg best_val_score']    
            tqdm_ids.set_postfix(best_train_loss = avg['avg best_train_loss'],
                            best_val_loss = avg['avg best_val_loss'],
                            best_test_loss = avg['avg best_test_loss'],
                            best_train_log10_MSE = avg['avg best_train_score'],
                            best_val_log10_MSE = avg['avg best_val_score'],
                            best_test_log10_MSE = avg['avg best_test_score'])    
    else:
        raise NotImplementedError()


    #res.update({'ray_id': id_})
    final_json.append(res)
    return df, tqdm_ids, final_json, best_score,ray_ids


def model_selection(model_name: str,
                    data_name: str,
                    early_stopping_patience: Optional[int] = None,
                    epochs: int = 1000,
                    task = None,
                    data_dir: str = '.',
                    exp_dir: str = '.',
                    dtype = torch.float32,
                    parallelism = float('inf')):
    """
    Perform a model selection phase through standard validation or k-fold model selection.
    All the results are saved into a DataFrame and the best configuration is returned.
    """

    assert ray.is_initialized() == True, "Ray is not initialized"
    data_dir = os.path.abspath(data_dir) # ray wants absolute paths
    exp_dir = os.path.abspath(exp_dir)

    assert not os.path.exists(join(exp_dir, 'results.csv')), 'The file results.csv already exists.'
    
    # Download data once for all configurations
    data_train, data_valid, data_test, num_features, num_classes = get_dataset(root=data_dir, name=data_name, task=task)
    del data_train, data_valid, data_test

    # Create the checkpoint directory
    checkpoint_dir = join(exp_dir, 'checkpoints')
    create_if_not_exist(checkpoint_dir)

    config_fun, model = CONFIGS[model_name]
    ray_ids = []
    ids_to_configs = {}
    
    if data_name == 'GraphProp':
        batch_size = 512
        seeds = [41, 95, 12, 35]
    else:
        batch_size = None
        seeds = None


    df = []
    final_json = []
    best_score = None
    tqdm_ids = tqdm.tqdm(total=sum([1 for _ in config_fun(num_features, num_classes)]))
    for conf_id, conf in enumerate(config_fun(num_features, num_classes)):
            conf.update({
                'exp':{'conf_id': conf_id,
                       'task': task,
                       'epochs': epochs,
                       'patience': early_stopping_patience,
                       'batch_size': batch_size,
                       'seeds': seeds}
            })
            conf['model'].update({
                'input_dim': num_features,
                'output_dim': num_classes,
                'node_level_task': task in NODE_LVL_TASKS
            })

            
            checkpoint_path = join(checkpoint_dir, f'conf_id_{conf_id}.pth')
            if data_name == 'GraphProp' in data_name:
                cpus_per_task = 1
                gpus_per_task = 1
                    
                ray_ids.append(
                    train_val_pipeline_GraphProp.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(model, conf, data_dir, data_name,
                                            early_stopping_patience=early_stopping_patience,
                                            path_save_best=checkpoint_path,dtype=dtype)
                )
            else:
                raise NotImplementedError(f'train_val_pipeline_{data_name} was not implemented!')

            ids_to_configs[ray_ids[-1]] = conf
            while len(ray_ids) >= parallelism:
                df, tqdm_ids, final_json, best_score,ray_ids = wait_and_collect(ray_ids,ids_to_configs, df, exp_dir, tqdm_ids, data_name, final_json, best_score)

    # Wait and collect results
    while len(ray_ids):
        df, tqdm_ids, final_json, best_score,ray_ids = wait_and_collect(ray_ids,ids_to_configs, df, exp_dir, tqdm_ids, data_name, final_json, best_score)
    
    json_path = join(exp_dir, 'complete_results.json')
    dump(final_json, json_path)

    df = pd.DataFrame(df)
    csv_path = join(exp_dir, 'results.csv')
    df.to_csv(csv_path, index=False)

    if data_name == 'GraphProp' or "toy" in data_name or "Peptides" in data_name:
        final_json.sort(key=lambda x: x['avg_res']['avg best_val_score']) # smaller values are the best
        df = df.sort_values('avg best_val_score', ascending=True, ignore_index=True)
    else:
        raise NotImplementedError()

    dump(final_json, json_path)
    df.to_csv(csv_path, index=False)
    
    return final_json[0]
