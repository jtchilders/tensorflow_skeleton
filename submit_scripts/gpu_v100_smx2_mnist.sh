#!/bin/bash
#COBALT -n 3
#COBALT -t 360
#COBALT -q gpu_v100_smx2
#COBALT -A datascience

echo [$SECONDS] setup conda environment
source /gpfs/jlse-fs0/projects/datascience/parton/mlcuda/mconda/setup.sh

echo [$SECONDS] setup local env vars
export OMP_NUM_THREADS=16
NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=4
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

if [ $PROCS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $PROCS ranks
   HOROVOD=--horovod
fi

mpirun -n $PROCS --hostfile $COBALT_NODEFILE \
   python main.py -c configs/mnist.json --logdir logdir/$COBALT_JOBID --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS $HOROVOD
