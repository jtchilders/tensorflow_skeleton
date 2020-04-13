#!/bin/bash
#COBALT -n 2
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience

echo [$SECONDS] setup conda environment
source /projects/datascience/parton/conda/miniconda3/latest/setup.sh

echo [$SECONDS] setup local env vars
export HYPERTHREADS_PER_CORE=1
export CORES_PER_NODE=64
export OMP_NUM_THREADS=$(( HYPERTHREADS_PER_CORE * $CORES_PER_NODE ))
export RANKS_PER_NODE=1
export RANKS=$(( $RANKS_PER_NODE * $COBALT_PARTSIZE ))

if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
fi

aprun -n $RANKS -N $RANKS_PER_NODE -d $OMP_NUM_THREADS -j $HYPERTHREADS_PER_CORE --cc depth \
   python main.py -c configs/ilsvrc.json --debug --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS $HOROVOD
