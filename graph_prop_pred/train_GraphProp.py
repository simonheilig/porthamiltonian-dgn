import os
import torch
import ray
import time
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from utils import get_dataset
from torch_geometric.nn import global_add_pool
from utils.pna_dataset import GRAPH_LVL_TASKS, NODE_LVL_TASKS
from torch_scatter import scatter
import gc
from sklearn.metrics import average_precision_score,mean_absolute_error



def getCurrentMemoryUsage():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())


def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


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

def train(model, optimizer, dataloader, criterion, device,dtype):
    model.train()
    epoch_loss = 0
    epoch_train_MSE = 0
    for iter, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch.x = batch.x.to(dtype)
        batch.y = batch.y.to(dtype)
        batch = batch.to(device)
        
        out = model(batch)
        loss = criterion(out, batch.y, batch.batch)
        loss_ = loss.detach().item()
        epoch_train_MSE += loss_
        
        loss.backward()
        optimizer.step()
       
        loss_ = loss.detach().item()
        epoch_loss += loss_
    
    epoch_loss /= (iter + 1)
    epoch_train_MSE /= (iter + 1)

    return epoch_loss, np.log10(epoch_train_MSE), optimizer

def train_sparse(model, optimizer, dataloader, criterion, device,dtype):
    model.train()
    epoch_loss = 0
    epoch_train_MSE = 0

    for iter, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch.x = batch.x.to(dtype)
        batch.y = batch.y.to(dtype)
        batch.adj_t = SparseTensor(row=batch.edge_index[0], col=batch.edge_index[1], value=batch.edge_attr)#
        batch = batch.to(device)

        out = model(batch.x,batch.adj_t,batch.batch,batch.ptr)
        loss = criterion(out, batch.y, batch.batch)
        loss_ = loss.detach().item()
        epoch_train_MSE += loss_

        loss.backward()
        optimizer.step()
       
        loss_ = loss.detach().item()
        epoch_loss += loss_
    
    epoch_loss /= (iter + 1)
    epoch_train_MSE /= (iter + 1)
    score = np.log10(epoch_train_MSE)

    return epoch_loss, score, optimizer

def evaluate(model, criterion, dataloader,device,dtype):
    model.eval()
    epoch_test_loss = 0
    epoch_test_MSE = 0
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            batch.x = batch.x.to(dtype)
            batch.y = batch.y.to(dtype)
            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y, batch.batch)
            loss_ = loss.detach().item()
            epoch_test_MSE += loss_
                    
                
            loss_ = loss.detach().item()
            epoch_test_loss += loss_
            
        epoch_test_loss /= (iter + 1)
        epoch_test_MSE /= (iter + 1)
        
    return epoch_test_loss, np.log10(epoch_test_MSE)

def evaluate_sparse(model, criterion, dataloader,device,dtype):
    model.eval()
    epoch_test_loss = 0
    epoch_test_MSE = 0
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            batch.x = batch.x.to(dtype)
            batch.y = batch.y.to(dtype)
            batch.adj_t = SparseTensor(row=batch.edge_index[0], col=batch.edge_index[1],value=batch.edge_attr )#, value=batch.edge_attr[:,0]
            batch = batch.to(device)

            out = model(batch.x,batch.adj_t,batch.batch,batch.ptr)
            loss = criterion(out, batch.y, batch.batch)
            loss_ = loss.detach().item()
            epoch_test_MSE += loss_
        
                
            loss_ = loss.detach().item()
            epoch_test_loss += loss_
            
        epoch_test_loss /= (iter + 1)
        epoch_test_MSE /= (iter + 1)
        score = np.log10(epoch_test_MSE)
        
    return epoch_test_loss, score

@ray.remote(num_cpus=1, num_gpus=1/4)
def train_val_pipeline_GraphProp(model_class, 
                           config, 
                           data_dir,
                           data_name,
                           early_stopping_patience=None, 
                           path_save_best=None, #eg, 'best_epoch_model.pth'
                           verbose=True,dtype=torch.float32):

    print('train', config)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # Load dataset
    data_train, data_valid, data_test, _, _ = get_dataset(root=data_dir, name=data_name, task=config['exp']['task'])
    
    results = []
    seeds = config['exp']['seeds']
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #eval('setattr(torch.backends.cudnn, "benchmark", False)')
        #eval('setattr(torch.backends.cudnn, "deterministic", True)')
        #eval('torch.use_deterministic_algorithms(True)')

        epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
        epoch_train_scores, epoch_val_scores, epoch_test_scores = [], [] , []
        per_epoch_time, per_epoch_mem = [], []
    
        epochs = config['exp']['epochs']
        best_score = None
        best_epoch = 0
        
        g_train = torch.Generator()
        g_train.manual_seed(0)

        model = model_class(**config['model'], dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=config['optim']['lr'], 
                                        weight_decay=config['optim']['weight_decay'])
            
        if os.path.exists(path_save_best.replace(".pth", f"_seed_{seed}.pth")):
            # Load the existing checkpoint
            print(f'Loading {path_save_best.replace(".pth", f"_seed_{seed}.pth")}')
            try:
                ckpt = torch.load(path_save_best.replace(".pth", f"_seed_{seed}.pth"), map_location=device)
                best_epoch = ckpt['epoch']
                best_score = ckpt['best_score']
                epoch_train_losses, epoch_val_losses, epoch_test_losses = ckpt['epoch_train_losses'], ckpt['epoch_val_losses'], ckpt['epoch_test_losses']
                epoch_train_scores, epoch_val_scores, epoch_test_scores = ckpt['epoch_train_scores'], ckpt['epoch_val_scores'], ckpt['epoch_test_scores']
                per_epoch_time = ckpt['per_epoch_time']
                per_epoch_mem = ckpt['per_epoch_mem']
                g_train.set_state(ckpt['g_train'].to(torch.device("cpu")))
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                optimizer_to(optimizer, device) # Map the optimizer to the current device

                if ckpt['train_ended']:
                    # The model was already trained, then return
                    results.append({
                        'best_train_loss': epoch_train_losses[best_epoch],
                        'best_val_loss': epoch_val_losses[best_epoch],
                        'best_test_loss': epoch_test_losses[best_epoch],
                        'best_train_score': epoch_train_scores[best_epoch],
                        'best_val_score': epoch_val_scores[best_epoch],
                        'best_test_score': epoch_test_scores[best_epoch],
                        'convergence time (epochs)': best_epoch,
                        'total time taken': sum(per_epoch_time),
                        'avg time per epoch': np.mean(per_epoch_time),
                        'avg mem per epoch' : np.mean(per_epoch_mem),
                        'model_params': sum(p.numel() for p in model.parameters())
                    })
                    continue
                best_epoch += 1
            except:
                print("This checkpoint is corrupted: ",path_save_best.replace(".pth", f"_seed_{seed}.pth"))


        node_level = config['exp']['task'] in NODE_LVL_TASKS
        assert node_level or config['exp']['task'] in GRAPH_LVL_TASKS
        def single_loss(pred, label, batch):
            # for node-level
            if node_level:
                nodes_in_graph = scatter(torch.ones(batch.shape[0]).to(device), batch).unsqueeze(1).to(device)
                #nodes_in_graph = torch.tensor([[(batch == i).sum()] for i in range(max(batch)+1)]).to(device)
                nodes_loss = (pred - label.reshape(label.shape[0], 1)) ** 2

                # Implementing global add pool of the node losses, reference here
                # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
                error = global_add_pool(nodes_loss, batch) / nodes_in_graph #average_nodes
                loss = torch.mean(error)
                return loss
            
            # for graph-level
            loss = torch.mean((pred - label.reshape(label.shape[0], 1)) ** 2)
            return loss
        

        criterion = single_loss
        model.to(device)

        #t0 = time.time()
        
        train_loader = DataLoader(data_train, batch_size=config['exp']['batch_size'], shuffle=True, generator=g_train)
        val_loader = DataLoader(data_valid, batch_size=config['exp']['batch_size'], shuffle=False)
        test_loader = DataLoader(data_test, batch_size=config['exp']['batch_size'], shuffle=False)

        for epoch in range(best_epoch, epochs):
                start = time.time()
                
                epoch_train_loss, epoch_train_score, optimizer = train(model, optimizer, train_loader, criterion, device,dtype)
                epoch_val_loss, epoch_val_score = evaluate(model, criterion, val_loader,device,dtype)

                epoch_test_loss, epoch_test_score = evaluate(model, criterion, test_loader, device,dtype)
                per_epoch_time.append(time.time()-start)
                per_epoch_mem.append(getCurrentMemoryUsage())
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_test_losses.append(epoch_test_loss)
                epoch_train_scores.append(epoch_train_score)
                epoch_val_scores.append(epoch_val_score)
                epoch_test_scores.append(epoch_test_score)

                # Record all statistics from this epoch
                if best_score is None or epoch_val_score <= best_score:
                    best_score = epoch_val_score
                    best_epoch = epoch
                    #print("New best epoch",epoch)
                    # Save model with highest evaluation score
                    if path_save_best is not None:
                        assert path_save_best[-4:] == ".pth", f'path_save_best should terminate with ".pth", received {path_save_best}'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_score': best_score,
                            'epoch_train_losses': epoch_train_losses,
                            'epoch_val_losses': epoch_val_losses, 
                            'epoch_test_losses': epoch_test_losses,
                            'epoch_train_scores': epoch_train_scores,
                            'epoch_val_scores': epoch_val_scores, 
                            'epoch_test_scores': epoch_test_scores,
                            'per_epoch_time': per_epoch_time,
                            'per_epoch_mem':per_epoch_mem,
                            'g_train': g_train.get_state(),
                            'train_ended': False,
                        }, path_save_best.replace(".pth", f"_seed_{seed}.pth"))

                if (early_stopping_patience is not None) and (epoch - best_epoch > early_stopping_patience):
                    print(config, f'-- seed: {seed}', f': early-stopped at epoch {epoch}')
                    break

                if epoch % 1 == 0:
                    print(np.mean(per_epoch_time), f' mean epoch time, {epoch}')
                    print(np.mean(per_epoch_mem), f' mean epoch mem, {epoch}')
                    if verbose:
                        print(f'Epochs: {epoch}, '
                            f'TR loss: {epoch_train_loss}, '
                            f'VL loss: {epoch_val_loss}, '
                            f'TR score: {epoch_train_score}, '
                            f'VL score: {epoch_val_score}, '
                            f'TEST score: {epoch_test_score}, '
                            f'lr: {optimizer.param_groups[0]["lr"]}')
                        

        ckpt = torch.load(path_save_best.replace(".pth", f"_seed_{seed}.pth"), map_location=device)
        ckpt['train_ended'] = True
        torch.save(ckpt, path_save_best.replace(".pth", f"_seed_{seed}.pth"))

        results.append({
            'best_train_loss': epoch_train_losses[best_epoch],
            'best_val_loss': epoch_val_losses[best_epoch],
            'best_test_loss': epoch_test_losses[best_epoch],
            'best_train_score': epoch_train_scores[best_epoch],
            'best_val_score': epoch_val_scores[best_epoch],
            'best_test_score': epoch_test_scores[best_epoch],
            'convergence time (epochs)': best_epoch, # epoch,
            'total time taken': sum(per_epoch_time), #time.time() - t0,
            'avg time per epoch': np.mean(per_epoch_time),
            'avg mem per epoch': np.mean(per_epoch_mem),
            'model_params': sum(p.numel() for p in model.parameters())
        })

        del model, optimizer
        gc.collect()
    
    avg = {}
    for k in results[0].keys():
        avg[f'avg {k}'] = np.mean([r[k] for r in results])
        avg[f'std {k}'] = np.std([r[k] for r in results])

    del data_train, data_valid, data_test
    gc.collect()
    return {'avg_res': avg, 'single_res': results}
