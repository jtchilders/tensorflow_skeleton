#!/bin/bash

# get current folder containing this script
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo DIR=$DIR
# get home directory without symobolic links
FULL_HOME=$( cd $HOME && pwd -LP)
echo FULL_HOME=$FULL_HOME
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

# MNIST expects the mnist.tgz file to be in $HOME/.keras/datasets
if [ ! -d $HOME/.keras/datasets/mnist.tgz ]; then
   mkdir -p $HOME/.keras/datasets
   cp $DIR/../datasets/mnist.tgz $HOME/.keras/datasets/
fi


CONTAINER=/home/parton/tensorflow-20.08-tf2-py3.simg

mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN  \
  singularity exec --nv -B $FULL_HOME  $CONTAINER  \
    python $DIR/../main.py -c $DIR/../configs/mnist.json \
    --logdir $DIR/../logdir/$COBALT_JOBID --intraop 16 --interop 16 --horovod
