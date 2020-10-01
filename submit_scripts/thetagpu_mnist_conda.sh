#!/bin/bash

source /lus/theta-fs0/projects/datascience/parton/conda/miniconda3/latest/setup.sh
export LD_LIBRARY_PATH=/lus/theta-fs0/projects/datascience/parton/cuda/TensorRT-7.2.0.14/lib:/lus/theta-fs0/projects/datascience/parton/cuda/cudnn-8.0.4/lib64::/lus/theta-fs0/projects/datascience/parton/cuda/nccl_2.7.8-1+cuda11.1_x86_64/lib
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
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

# MNIST expects the mnist.tgz file to be in $HOME/.keras/datasets
if [ ! -d $HOME/.keras/datasets/mnist.tgz ]; then
   mkdir -p $HOME/.keras/datasets
   cp $DIR/../datasets/mnist.tgz $HOME/.keras/datasets/
fi

echo which mpirun = $(which mpirun)
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN  \
    python $DIR/../main.py -c $DIR/../configs/mnist.json \
    --logdir $DIR/../logdir/$COBALT_JOBID --intraop 16 --interop 16 --horovod
