#!/bin/bash
#COBALT -n 1
#COBALT -t 360
#COBALT -q dgx
#COBALT -A datascience

echo [$SECONDS] setup conda environment
#source /gpfs/jlse-fs0/projects/datascience/parton/mlcuda/mconda/setup.sh
source /gpfs/jlse-fs0/projects/datascience/parton/conda/2020-06/setup.sh

echo [$SECONDS] setup local env vars
export OMP_NUM_THREADS=32
export RANKS=8

if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
fi

mpirun -n $RANKS \
   python main.py -c configs/ilsvrc.json --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS $HOROVOD
