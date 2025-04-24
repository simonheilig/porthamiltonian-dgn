## Graph Property Prediction Experiment
We consider the prediction of three graph properties - Diameter, Single-Source Shortest Paths (SSSP), and node Eccentricity on synthetic graphs following the setup outlined in [_Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023_](https://github.com/gravins/Anti-SymmetricDGN/tree/main/graph_prop_pred).

## How to reproduce our results
1) In the file ```run_all.sh```:
- set the gpu ids. Default is ```0```.
- set the variable ```save_dir```, i.e., root directory that stores the ```data``` folder and the results. Default is ```./save_dir_GraphProp/```.
- specify the model configuration corresponding to ```conf.py```
2) unpack ```data.tar.gz``` into the specified ```$save_dir/data``` folder
3) Run: ``` ./run_all.sh ```