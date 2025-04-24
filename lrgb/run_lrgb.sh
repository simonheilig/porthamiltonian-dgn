#!/usr/bin/env bash

CONFIG=${CONFIG:-peptides-func-phdgn} # peptides-struct-phdgn, peptides-func-phdgn-conservative, peptides-struct-phdgn-conservative
GRID=${GRID:-peptides-phdgn} # peptides-phdgn-conservative
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-100}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}

# generate configs 
python configs_gen.py --config configs/PHDGN/${CONFIG}.yaml \
  --grid grids/PHDGN/${GRID}.txt \
  --out_dir configs/grid

# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/grid/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch, check on the metric corresponding to the task (struct = mae, func = ap)
python agg_batch.py --dir results/${CONFIG}_grid_${GRID} --metric ap
