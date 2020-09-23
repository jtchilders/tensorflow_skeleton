#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MPIPATH=/usr/mpi/gcc/openmpi-4.0.3rc4

echo COBALT_NODEFILE=$COBALT_NODEFILE
echo COBALT_JOBID=$COBALT_JOBID

export LD_LIBRARY_PATH=$MPIPATH/lib:$LD_LIBRARY_PATH
export PATH=$MPIPATH/bin:$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PATH=$PATH

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

CONTAINER=/home/parton/tensorflow-20.08-tf2-py3.simg

mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN  \
  singularity exec --nv $CONTAINER  \
    python $DIR/../main.py -c $DIR/../configs/mnist.json \
    --logdir logdir/$COBALT_JOBID --intraop 16 --interop 16 --horovod
