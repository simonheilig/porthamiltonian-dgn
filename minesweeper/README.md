## Minesweeper Experiment
We consider the Minesweeper Dataset from [_Platanov et al. A critical look at the evaluation of GNNs under heterophily: Are we really making progress?. ICLR 2023_](https://github.com/yandex-research/heterophilous-graphs) under the training protocol of [_Luo et al. Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification. NeurIPS 2024 Track Datasets and Benchmarks_](https://github.com/LUOyk1999/tunedGNN).

## How to reproduce our results
1) In the file ```run_minesweeper.sh```:
- set the gpu ids. Default is ```0```.
- set the variable ```save_dir```, i.e., root directory that stores the ```data``` folder and the results. Default is ```./saving_dir/```.
- specify the model configuration corresponding to ```conf.py```
2) Run: ``` ./run_minesweeper.sh ```