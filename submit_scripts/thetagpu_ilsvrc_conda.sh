#!/bin/bash
#COBALT -O logdir/$COBALT_JOBID

MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3
source $MCONDA/setup.sh

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

EXEC=$(which python)
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE $(which python)"
fi

export OMP_NUM_THREADS=64
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PATH=$PATH
echo which python = $(which python)
LOGDIR=logdir/$COBALT_JOBID/$(date +"%Y-%m-%d-%H-%M")/conda
mkdir -p $LOGDIR
echo $LOGDIR
cp $0 $LOGDIR/
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
#export TF_XLA_FLAGS=--tf_xla_auto_jit=1
#export TF_XLA_FLAGS=--tf_xla_auto_jit=fusible
$EXEC -c 'import os;print(os.environ["PATH"])'
$EXEC main.py -c configs/ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
   --logdir $LOGDIR $HOROVOD --profiler --batch-term 50 
