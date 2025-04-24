from models import *
from utils import cartesian_product


def get_PHDGN_conservative_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers':[1],
        'activation':['tanh'],
        'num_iters': [distance * ni for ni in range(1,4)],
        'double_dim': [False, True],
        'pq': ['p', 'q', 'pq'],
        'epsilon': [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 1e-4],
        'p_conv_mode': ['naive', 'gcn'],
        'q_conv_mode': ['naive', 'gcn']
    }
    for params in cartesian_product(grid):
        if params['p_conv_mode']=='gcn' and params['q_conv_mode']=='naive':
            continue
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_PHDGN_conf(in_channels, distance):
    grid = {
        'alpha': [0., 1.], # no external force/dampening
        'beta': [0., 1.], # no external force/dampening
        'dampening_mode': ['param'],
        'external_mode': ['DGNtanh']
    }
    for params in get_PHDGN_conservative_conf(in_channels, distance):
        for conf in cartesian_product(grid):
            if conf['alpha'] == 0 and conf['beta'] == 0: 
                continue ## in this case we are back to conservative PHDGN
            params.update(conf)
            yield params
    
MODELS = {
    'phdgn': (PHDGN_Model, get_PHDGN_conf),
    'phdgn_conservative': (PHDGN_Model, get_PHDGN_conservative_conf)
}