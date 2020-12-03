#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# get home directory without symobolic links
FULL_HOME=$( cd $HOME && pwd -LP)
echo FULL_HOME=$FULL_HOME

MPIPATH=/usr/mpi/gcc/openmpi-4.0.3rc4
echo MPIPATH=$MPIPATH

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

EXEC=python
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE python"
fi

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs

export OMP_NUM_THREADS=64
#CONTAINER=/home/parton/tensorflow-20.08-tf2-py3.simg
CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg
#CONTAINER=/lus/theta-fs0/projects/datascience/parton/thetagpu/singularity/tf2/tf2_20.10-py3_imagenet.simg
#CONTAINER=/raid/scratch/tf2_20.10-py3_imagenet.simg
#HOSTS=$(cat $COBALT_NODEFILE | sed ':a;N;$!ba;s/\n/,/g')
LOGDIR=$DIR/../logdir/$COBALT_JOBID/$(date +"%Y-%m-%d-%H-%M")/cont
mkdir -p $LOGDIR
echo LOGDIR=$LOGDIR
cp $0 $LOGDIR/
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
export TF_XLA_FLAGS=--tf_xla_auto_jit=1
export TF_XLA_FLAGS=--tf_xla_auto_jit=fusible
singularity exec --nv -B /lus/theta-fs0/software/thetagpu -B $FULL_HOME \
   -B /lus/theta-fs0/projects/datascience/parton/ \
   -B /lus/theta-fs0/software/datascience/ $CONTAINER \
      $EXEC /home/parton/git/tensorflow_skeleton/main.py -c $DIR/../configs/ilsvrc.json \
            --logdir $DIR/../$LOGDIR --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS \
            $HOROVOD  > $LOGDIR/output.txt 2>&1 #--profiler --batch-term 50
