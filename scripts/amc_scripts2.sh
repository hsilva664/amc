#!/bin/bash

#SBATCH -p mmaire-gpu --exclude=gpu-g6 -c 1 -J h_amp_search -d singleton --array=1-24

bash -c ". ~/torchenv/bin/activate; sh ~/begin.sh; `sed "${SLURM_ARRAY_TASK_ID}q;d" scripts/amp_scripts2.txt`"