#!/bin/bash
#COBALT -n 2
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT -O logdir/$COBALT_JOBID

echo [$SECONDS] setup conda environment
module load miniconda-3/2020-12

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

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

#export MKL_VERBOSE=1
#export DNNL_VERBOSE=1
#export ONEDNN_VERBOSE=1
#export dnnl_verbose=1
#-d $OMP_NUM_THREADS -j $HYPERTHREADS_PER_CORE --cc depth
env | sort > logdir/$COBALT_JOBID.env
aprun -n $RANKS -N $RANKS_PER_NODE --cc none \
   python main.py -c configs/ilsvrc.json --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS --profiler --batch-term 50 $HOROVOD

