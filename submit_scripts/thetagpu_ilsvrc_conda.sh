#!/bin/bash

#MCONDA=/lus/theta-fs0/projects/datascience/parton/thetagpu/tf-build
MCONDA=/lus/theta-fs0/projects/datascience/parton/thetagpu/tf-build3/tf-intall/mconda3
#source /lus/theta-fs0/projects/datascience/parton/thetagpu/tf-build/miniconda3/latest/setup.sh
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
LOGDIR=logdir/$COBALT_JOBID/$(date +"%Y-%M-%d-%H")/conda
mkdir -p $LOGDIR
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
$EXEC main.py -c configs/ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS --logdir $LOGDIR $HOROVOD #--profiler --batch-term 50
