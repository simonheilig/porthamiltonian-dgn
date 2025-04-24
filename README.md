# Port-Hamiltonian DGN
This repository provides the official reference implementation of our paper
**[Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks](https://openreview.net/forum?id=03EkqSCKuO)** published at ICLR 2025.

Please consider citing us

 	@inproceedings{
        heilig2025porthamiltonian,
        title={Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks},
        author={Simon Heilig and Alessio Gravina and Alessandro Trenta and Claudio Gallicchio and Davide Bacciu},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025}
    }


## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

1. Install the required packages and create the environment
    - ``` conda env create -f env.yml ```
    - python=3.8.18, pyg=2.4.0, pytorch=2.1.0, CUDA 12.1

2. Activate the environment
    - ``` conda activate phdgn ```

## Run the experiment
To reproduce the experiments please refer to:
- [graph_prop_pred/README.md](https://github.com/simonheilig/porthamiltonian-dgn/tree/main/graph_prop_pred) to run the diameter, SSSP, and eccentricity experiments.
- [graph_transfer/README.md](https://github.com/simonheilig/porthamiltonian-dgn/tree/main/graph_transfer_task) to run the graph transfer tasks on line, ring, and crossed-ring topologies.
- [lrgb/README.md](https://github.com/simonheilig/porthamiltonian-dgn/tree/main/lrgb) to run the LRGB experiments.
- [minesweeper/README.md](https://github.com/simonheilig/porthamiltonian-dgn/tree/main/minesweeper) to run the Minesweeper experiments.

## Repository structure
The repository is structured as follows:

    ├── README.md                  <- The top-level README.
    │
    ├── env.yml                    <- The conda environment requirements.
    │
    └── graph_prop_pred            <- Contains the code to reproduce the graph property prediction experiment.
    │   ├── run_all.sh             <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
    │   ├── models                 <- Contains the code for the framework PH-DGN
    │   ├── conf.py                <- Contains the hyper-parameter space for the model.
    │   └── main.py                <- The main.
    │
    └── graph_transfer_task        <- Contains the code to reproduce the graph transfer experiment.
    │   ├── run_graph_transfer.sh  <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
    │   ├── models                 <- Contains the code for the framework PH-DGN
    │   ├── conf.py                <- Contains the hyper-parameter space for the model.
    │   └── main.py                <- The main.
    │
    └── lrgb                       <- Contains the code to reproduce the long-range graph benchmark experiment.
    │   ├── run_lrgb.sh            <- The script used to run the experiment. Note: you need to specify the name of the base model configuration and corresponding grid
    │   ├── configs                <- Contains the base model configuration for graphgym.
    │   ├── custom_graphgym        <- Contains the extension of graphgym from Teonshoff et al. and the PH-DGN layer.
    │   ├── grids                  <- Contains the hyper-parameter space for the model.
    │   └── main.py                <- The main.
    │
    └── minesweeper                <- Contains the code to reproduce the minesweeper experiment.
        ├── run_minesweeper.sh     <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
        ├── models                 <- Contains the code for the framework PH-DGN
        ├── conf.py                <- Contains the hyper-parameter space for the model.
        └── main.py                <- The main.