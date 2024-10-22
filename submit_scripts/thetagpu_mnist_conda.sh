#!/bin/bash
#COBALT -n 1
#COBALT -t 30
#COBALT -A datascience
#COBALT -O logdir/$COBALT_JOBID

MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2021-01-08/mconda3
source $MCONDA/setup.sh

# get current folder containing this script
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo DIR=$DIR
# get home directory without symobolic links
FULL_HOME=$( cd $HOME && pwd -LP)
echo FULL_HOME=$FULL_HOME
#MPIPATH=/usr/mpi/gcc/openmpi-4.0.3rc4


echo COBALT_NODEFILE=$COBALT_NODEFILE
echo COBALT_JOBID=$COBALT_JOBID

export LD_LIBRARY_PATH=$MPIPATH/lib:$LD_LIBRARY_PATH
export PATH=$MPIPATH/bin:$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PATH=$PATH

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS
export OMP_NUM_THREADS=8

# MNIST expects the mnist.tgz file to be in $HOME/.keras/datasets
if [ ! -d $HOME/.keras/datasets/mnist.tgz ]; then
   mkdir -p $HOME/.keras/datasets
   cp $DIR/../datasets/mnist.tgz $HOME/.keras/datasets/
fi

EXEC=$(which python)
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE $(which python)"
fi

export TF_XLA_FLAGS=--tf_xla_auto_jit=1

$EXEC main.py -c $DIR/../configs/mnist.json \
    --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS $HOROVOD
