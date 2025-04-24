
export CUDA_VISIBLE_DEVICES=0
num_gpus=1 # the percentage of gpu per configuration
num_cpus=5 # the number of cpus per configuration
dist=50 # the distance considered in the experiment. Choose between 50, 10, 5, and 3
exp_root=./exp_root
gpu_count=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c) + 1 ))
p=$(echo "$gpu_count / $num_gpus" | bc) # how many tasks I want to run in parallel. This is still bounded by the number of available gpus and the percentage of used gpu per config

m=phdgn_conservative # the model selected for the evaluation. Choose between phdgn_conservative and phdgn
python3 -u main.py --m $m --batch 512 --ngpus $num_gpus --ncpus $num_cpus --distance $dist --root $exp_root --epochs 2000 --parallelism $p #> ./$m\_$dist\_out 2> ./$m\_$dist\_err
