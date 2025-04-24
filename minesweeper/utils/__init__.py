from os import remove
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix,average_precision_score,mean_absolute_error,roc_auc_score
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Actor, WikipediaNetwork, WebKB, LRGBDataset
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import subgraph
import torch_geometric.transforms as T
from collections import defaultdict
from typing import Tuple, Optional

def set_seed(seed):
    import torch
    import random
    import numpy as np

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scoring(y_true, y_pred, labels=None):
    s = {}
    for k in SCORE:
        s[k] = (SCORE[k](y_true, y_pred, average='macro') if k != 'accuracy'
                else SCORE[k](y_true, y_pred))

    s["confusion_matrix"] = None#,confusion_matrix(y_true, y_pred, labels=labels)
    return s


def get_dataset(root:str, name:str) -> Tuple:
    data_getter = DATA[name]
    if name == 'Actor':
        data = data_getter(root=root)
    elif name in ['Squirrel', 'Chameleon']:
        name = name.lower()
        data = data_getter(root=root, name=name, geom_gcn_preprocess=True)
    #elif name in []:
    #    data = data_getter(root=root, name=name, transform=T.ToUndirected())[0]
    else:
        data = data_getter(root=root, name=name)
    
    num_features = data.num_features
    num_classes = data.num_classes
    data = data[0]

    return data, num_features, num_classes
    

DATA = {
    'Squirrel': WikipediaNetwork,
    'Chameleon': WikipediaNetwork,
    'Actor': Actor,
    'Texas': WebKB,
    'Cornell': WebKB,
    'Wisconsin': WebKB,
    'Roman-empire':HeterophilousGraphDataset,
    'Amazon-ratings':HeterophilousGraphDataset,
    'Minesweeper':HeterophilousGraphDataset,
    'Tolokers':HeterophilousGraphDataset,
    'Questions':HeterophilousGraphDataset,
    'Peptides-func':LRGBDataset,
    'Peptides-struct':LRGBDataset,
}

SCORE = {
 #"f1_score": f1_score,
 #"recall": recall_score,
 "accuracy": roc_auc_score,
 #"average_precision":average_precision_score,
 #"mae":mean_absolute_error,
 "roc_auc":roc_auc_score,
}
