# Graph Transfer Experiment
This task (proposed in [1]) consists on transfering a label from a source node to a target node on different graph topologies.

[1] Gravina et al. "On Oversquashing in Graph Neural Networks Through The Lens of Dynamical Systems". In AAAI 2025

## How to reproduce the experiments

To reproduce our results please:
1) Download and uncompress [```data.zip```](https://github.com/gravins/SWAN/blob/main/graph_transfer/data.zip)
2) In the file ```run_transfer_graph.sh``` set:
- set the variable ```exp_root```, i.e., root directory that stores the ```data``` and ```results``` folders
- set the ```CUDA_VISIBLE_DEVICES```
- set the percentage of gpu used by configuration (```num_gpus```) and the number of cpus (```num_cpus```) used by configuration in the experiment
- set the distance considered in the experiment (```dist```) 
3) Run: ``` run_transfer_graph.sh ```


Since this folder is a clone of [SWAN/graph_transfer](https://github.com/gravins/SWAN/tree/main/graph_transfer) with the only addition of the new PH-DGN model, we refer the user to the original repository for further details and specifications.