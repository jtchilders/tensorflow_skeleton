#!/bin/bash
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

#HOSTS=$(cat $COBALT_NODEFILE | sed ':a;N;$!ba;s/\n/,/g')
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN  singularity exec --nv -B /lus/theta-fs0/software/thetagpu -B /lus/theta-fs0/projects/datascience/parton/image_data:/projects/datascience/parton/image_data   /home/parton/tensorflow-20.08-tf2-py3.simg python /home/parton/git/tensorflow_skeleton/main.py -c /home/parton/git/tensorflow_skeleton/configs/ilsvrc.json --logdir logdir/$COBALT_JOBID --intraop 16 --interop 16 --horovod
