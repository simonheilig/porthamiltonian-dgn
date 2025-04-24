import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from utils.io import join
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np

def transform(node_labels, graph_labels):
    # normalize labels
    max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
    max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
    for dset in node_labels.keys():
        node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
        graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]
    
    return node_labels, graph_labels

TASKS = ['dist', 'ecc', 'lap', 'conn', 'diam', 'rad','toy','pepfunc','pepstruct']
NODE_LVL_TASKS = ['dist', 'ecc', 'lap','toy']
GRAPH_LVL_TASKS = ['conn', 'diam', 'rad','pepfunc','pepstruct']

class GraphPropDataset(InMemoryDataset):
    def __init__(self, root, split, task, dim='25-35', pre_transform=None):
        assert split in ['train', 'val', 'test']
        assert task in TASKS
        if not task in ['dist', 'ecc', 'diam']:
            raise NotImplementedError('the only tasks implemented are: dist, ecc, diam')

        assert dim in ['15-25', '25-35']
        self.dim = dim

        self.split = split
        self.task = task
        super().__init__(root)
        self.pre_transform = pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f'Loaded processed {self.processed_paths[0]}')

    @property
    def processed_file_names(self):
        return [join(self.root, f'{self.split}_{self.task}_{self.dim}_data.pt')]

    def process(self):
        (adj, features, 
         node_labels, graph_labels) = torch.load(open(join(self.root, f'pna_dataset_{self.dim}.pkl'),'rb'))

        # node_labels ["eccentricity", "graph_laplacian_features", "sssp"]
        # graph_labels ["is_connected", "diameter", "spectral_radius"]

        if self.pre_transform is not None:
            node_labels, graph_labels = self.pre_transform(node_labels, graph_labels)

        data_list = []
        n_batches = len(adj[self.split])
        ns = []
        degs = []
        diams = []
        edges = []
        density = []
        shortest = []
        ng = 0
        for batch_id in range(n_batches):
            n_samples_in_batch = len(adj[self.split][batch_id])
            for sample_id in range(n_samples_in_batch):
                
                a = adj[self.split][batch_id][sample_id]
                ft = features[self.split][batch_id][sample_id]
                nl = node_labels[self.split][batch_id][sample_id]
                gl = graph_labels[self.split][batch_id][sample_id]
                
                edge_index, edge_attr = dense_to_sparse(a)
                if self.task == 'dist':
                    y = nl[:, 2]
                elif self.task == 'ecc':
                    y = nl[:, 0]
                elif self.task == 'diam':
                    y = gl[1]
                else:
                    raise NotImplementedError()

                d = Data(x=ft, edge_index=edge_index, y=y)
                g = to_networkx(d,to_undirected=True)
                #print(g.number_of_nodes(),nx.is_connected(g))
                if nx.is_connected(g):
                    diam = nx.diameter(g)
                    diams.append(diam)
                ns.append(g.number_of_nodes())
                degs.append(sum(dict(g.degree()).values()) / g.number_of_nodes())
                edges.append(g.number_of_edges())
                density.append(nx.density(g))
                ng+=1
                data_list.append(
                    d
                )
        print("split {}, #graphs {}, min node{}, avg node {}, max node {}, min degree {}, avg degree {}, max degree {}, min diameter {}, avg diameter {}, max diameter {}, min edges {}, avg edges {}, max edges {}, min density {}, avg density {}, max density {}".format(self.split,ng,np.min(ns),np.mean(ns),np.max(ns),np.min(degs),np.mean(degs),np.max(degs),np.min(diams),np.mean(diams),np.max(diams),np.min(edges),np.mean(edges),np.max(edges),np.min(density),np.mean(density),np.max(density)))
        data, slices = self.collate(data_list)
        print(f'Loaded {self.processed_paths[0]}')
        torch.save((data, slices), self.processed_paths[0])
