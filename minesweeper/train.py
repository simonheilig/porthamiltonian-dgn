from unittest import result
import torch
import ray
import time
import random
import datetime
import numpy as np
from conf import CONFIGS
from utils import scoring, get_dataset
from models import *
import os
from torch_sparse import SparseTensor
from torch.amp import autocast, GradScaler

def calculate_spectral_norm(weight_matrix):
    # Power iteration method to estimate the spectral norm
    u = torch.randn(1, weight_matrix.size(1)).to(weight_matrix.device)
    v = torch.randn(1, weight_matrix.size(0)).to(weight_matrix.device)
    sigma = 0

    for _ in range(100):
        v = torch.nn.functional.normalize(torch.matmul(u, weight_matrix), p=2)
        u = torch.nn.functional.normalize(torch.matmul(v, weight_matrix.t()), p=2)
        sigma = torch.mm(u, torch.mm(weight_matrix, v.t()))

    return sigma[0,0]

def optimizer_to(optim, device):
    # Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train(model, optimizer,scheduler, scaler, data, train_mask, criterion,device=None):
    model.train()
    # Reset gradients from previous step
    model.zero_grad()

    data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=data.edge_attr)#
    data = data.to(device)
    with autocast(enabled=False,device_type=data.x.device.type):
        # Perform a forward pass
        preds = model.forward(data,data.adj_t)
        # Perform a backward pass to calculate the gradients
        loss = criterion(preds[train_mask].flatten(), data.y[train_mask].float())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # Update parameters
    optimizer.zero_grad()
    scheduler.step()


def evaluate(model, data, eval_mask, criterion, return_true_values=False,num_classes=0,device=None):
    t0 = time.time()
    model.eval()
    y_true, y_preds, y_preds_confidence = [], [], []
    with torch.no_grad():
        data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=data.edge_attr)#
        data = data.to(device)

        # Perform a forward pass
        preds = model.forward(data,data.adj_t)[eval_mask].flatten()
        y_true = data.y[eval_mask]

        loss = criterion(preds, y_true.float())
               
        y_preds = preds.cpu().tolist()
        y_true = y_true.cpu().tolist()
        if return_true_values:
            preds = (torch.sigmoid(preds) > 0.5).float()
            y_preds_confidence = preds.cpu().tolist()

    scores = {'loss': loss.cpu().item(),
              'time': datetime.timedelta(seconds=time.time() - t0)}

    # Compute scores
    scores.update(scoring(y_true, y_preds,labels=range(num_classes)))
    true_values = (y_true, y_preds, y_preds_confidence) if return_true_values else None
    return scores, true_values


@ray.remote(num_cpus=2, num_gpus=0.25)
def train_and_eval(model_class, config, 
                   data_dir, data_name,
                   metric='accuracy', device="cpu", mode="Validation", early_stopping_patience=None, 
                   path_save_best=None, #eg, 'best_epoch_model.pth'
                   verbose=True):
    path_best = path_save_best

    # Load dataset
    data, _, num_c = get_dataset(root=data_dir, name=data_name)
    epochs = config['exp']['epochs']

    print('train', config)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    result_per_split = []

    for i in range(10):
        # training and evaluation on the i-th TR/VL/TS split
        total_time = time.time()

        tr_mask = data.train_mask.T[i]
        eval_mask = data.val_mask.T[i]
        test_mask = data.test_mask.T[i]

        model = model_class(**config['model'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'], weight_decay=config['optim']['wd'])
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = config['optim']['lr']
        def get_lr_multiplier(step):
            if step < 1:
                return (step + 1) / (1 + 1)
            else:
                return 1
        scaler = GradScaler(enabled=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=epochs)
        max_score = -1
        best_epoch = 0
        best_score = None
        epochs = config['exp']['epochs']
        path_save_best = path_best + f'_{i}.pth'
        if os.path.exists(path_save_best):
            # Load the existing checkpoint
            print(f'Loading {path_save_best}')
            ckpt = torch.load(path_save_best, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer_to(optimizer, device) # Map the optimizer to the current device
            max_score = ckpt['max_score']
            best_epoch = ckpt['epoch']
            best_score = ckpt['best_score']
            epochs = epochs - ckpt['epoch']

        model.to(device)
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
        data = data.to(device)
        for e in range(epochs):
            t0 = time.time()
            data = data.to(device)
            train(model, optimizer,scheduler,scaler, data, tr_mask, criterion,device)
            
            # Evaluate the model on the training set
            train_scores, _ = evaluate(model, data, tr_mask, criterion,num_classes=num_c,device=device)
            tr_time = datetime.timedelta(seconds=time.time() - t0)
            
            # Evaluate the model on the evaluation set
            eval_scores, _ = evaluate(model, data, eval_mask, criterion,num_classes=num_c,device=device)

            # Record all statistics from this epoch
            train_scores['time'] = tr_time
            h = {'epoch': e + 1,
                'Training': train_scores,
                mode: eval_scores}


            if eval_scores[metric] >= max_score:
                max_score = eval_scores[metric]
                best_epoch = e
                best_score = h

                # Save model with highest evaluation score
                if path_save_best is not None:
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_score': best_score,
                        'max_score': max_score
                        }, path_save_best)

            if verbose:
                print(f'Epochs: {e}, '
                    f'TR Loss: {train_scores["loss"]} '
                    f'VL Loss:{eval_scores["loss"]} '
                    f'TR Acc: {train_scores["accuracy"]} '
                    f'VL Acc:{eval_scores["accuracy"]}'
                    f'TR Roc: {train_scores["roc_auc"]} '
                    f'VL Roc:{eval_scores["roc_auc"]}')

            if (early_stopping_patience is not None) and (e - best_epoch > early_stopping_patience):
                break

        # Evaluate the model on the test set
        ckpt = torch.load(path_save_best, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        test_scores, _ = evaluate(model, data, test_mask, criterion,num_classes=num_c)

        m = 'VL' if mode == 'Validation' else 'TS'
        print(config, f" - Total training took {datetime.timedelta(seconds=time.time()-total_time)} (hh:mm:ss)",
            f'Training stopped after {e} epochs',
            f'*** Best Epoch: {best_epoch}, '
            f'TR Loss: {best_score["Training"]["loss"]}'
            f'{m} Loss:{best_score[mode]["loss"]}'
            f'Test Loss:{test_scores["loss"]}'
            f'TR Acc: {best_score["Training"]["accuracy"]}'
            f'{m} Acc:{best_score[mode]["accuracy"]}'
            f'Test Acc:{test_scores["accuracy"]}')

        result_per_split.append({
            'best_epoch': best_epoch,
            'Training':  best_score["Training"],
            mode: best_score[mode],
            'Test': test_scores,
        })
    
    return result_per_split


@ray.remote(num_cpus=1, num_gpus=1)
def eval_test(data_dir, data_name, metric, 
              best_conf, model_name, checkpoints_paths, device):
    
    avg_ts_score = {}
    m, l = [], []
    for data_seed in best_conf.keys():
        # Load dataset
        data, num_features, num_classes = get_dataset(root=data_dir, name=data_name, seed=int(data_seed))
        ts_mask = data.test_mask

        paths = [p for p in checkpoints_paths if f'data_seed_{data_seed}' in p] # this contains the different initializations of the best conf for a particular TR/VL/TS split 

        ts_score = {'seed': [], 'loss':[], metric:[], 'conf_mat':[]}
        for ckpt_path in paths:
            model = CONFIGS[model_name][1]
            model = model(**best_conf[data_seed]['model'])

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            scores, _ = evaluate(model = model, 
                                data = data.to(device),
                                eval_mask = ts_mask,
                                criterion = torch.nn.CrossEntropyLoss(),num_classes=num_classes)
            seed = ckpt_path.split('_seed_')[-1].replace('.pth', '')
            ts_score['seed'].append(seed)
            ts_score[metric].append(scores[metric])
            ts_score['loss'].append(scores['loss'])
            ts_score['conf_mat'].append(str(scores['confusion_matrix'].tolist()))
            print(f'data seed {data_seed}, seed {seed}, confusion matrix {scores["confusion_matrix"]}')

        avg_ts_score[data_seed] = ts_score
        m.append(np.mean(ts_score[metric]).item())
        l.append(np.mean(ts_score['loss']).item())

    avg_ts_score[f'avg test {metric}'] = np.mean(m).item()
    avg_ts_score[f'std test {metric}'] = np.std(m).item()
    avg_ts_score['avg test loss'] = np.mean(l).item()
    avg_ts_score['std test loss'] = np.std(l).item()

    return avg_ts_score
