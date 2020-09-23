#!/bin/bash
#COBALT -n 3
#COBALT -t 360
#COBALT -q gpu_v100_smx2
#COBALT -A datascience

echo [$SECONDS] setup conda environment
source /gpfs/jlse-fs0/projects/datascience/parton/mlcuda/mconda/setup.sh

echo [$SECONDS] setup local env vars
export OMP_NUM_THREADS=32
export RANKS=3

if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
fi

mpirun -n $RANKS \
   python main.py -c configs/ilsvrc.json --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS $HOROVOD
