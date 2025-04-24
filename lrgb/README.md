# Long-Range Graph Benchmark Experiment
This experimental setup is based on the re-evaluation of [toenshoff/LRGB](https://github.com/toenshoff/LRGB/) for predicting peptides-func and peptides-struct.
Please check your python requirements for running the graphgym.
Note, the code relied on torchmetrics=0.7.2 and scikit-learn=1.3.0.

## How to reproduce our results
1) In the file ```run_lrgb.sh```:
- set the base configuration variable ```$CONFIG```
- set the hyperparameter space grid variable ```$GRID```
- set the number repetitions variable ```$REPEAT```
2) In the file ```parallel.sh```:
- set the GPU ids
3) Run the grid experiments: ``` ./run_lrgb.sh ```
4) Alternatively, start single configs:
``` python main.py --cfg configs/PHDGN/$CONFIG.yaml --repeat 3 ```



