#!/usr/bin/env python
import argparse,logging,json,time,os,sys
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import data_handler
import model,lr_func,losses,accuracies

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()
DEFAULT_LOGDIR = '/tmp/tf-' + str(os.getpid())

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def main():
   ''' simple starter program for tensorflow models. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',type=int,help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',type=int,help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('-l','--logdir',default=DEFAULT_LOGDIR,help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('--horovod', default=False, action='store_true', help="Use MPI with horovod")
   parser.add_argument('--profiler',default=False, action='store_true', help='Use TF profiler, needs CUPTI in LD_LIBRARY_PATH for Cuda')
   parser.add_argument('--profrank',default=0,type=int,help='set which rank to profile')

   parser.add_argument('--precision',default='float32',help='set which precision to use; options include: "float32","mixed_float16","mixed_bfloat16"')

   parser.add_argument('--batch-term',dest='batch_term',type=int,help='if set, terminates training after the specified number of batches',default=0)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   tf.config.threading.set_inter_op_parallelism_threads(args.interop)
   tf.config.threading.set_intra_op_parallelism_threads(args.intraop)

   hvd = None
   rank = 0
   nranks = 1
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   if args.horovod:
      import horovod
      import horovod.tensorflow as hvd
      hvd.init()
      logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + (
                 '%05d' % hvd.rank()) + ':%(name)s:%(message)s'
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         logging_level = logging.WARNING
      os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
      os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)
   if hvd:
      logging.warning('rank: %5d   size: %5d  local rank: %5d  local size: %5d', hvd.rank(), hvd.size(),
                      hvd.local_rank(), hvd.local_size())

   gpus = tf.config.experimental.list_physical_devices('GPU')
   logger.info('gpus = %s',gpus)
   if gpus:
      try:
         for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
         # Visible devices must be set before GPUs have been initialized
         raise
   
   # setting mixed precision policy
   # policy = mixed_precision.Policy(args.precision)
   # mixed_precision.set_policy(policy)

   # logger.info('device_str:                 %s', device_str)

   if 'CUDA_VISIBLE_DEVICES' in os.environ:
      logging.info('CUDA_VISIBLE_DEVICES=%s',os.environ['CUDA_VISIBLE_DEVICES'])
      logging.debug(device_lib.list_local_devices())
   else:
      logging.info('CUDA_VISIBLE_DEVICES not defined in os.environ')
   logging.info('using tensorflow version:   %s',tf.__version__)
   logging.info('using tensorflow from:      %s',tf.__file__)
   if hvd:
      logging.info('using horovod version:      %s',horovod.__version__)
      logging.info('using horovod from:         %s',horovod.__file__)
   logging.info('logdir:                     %s',args.logdir)
   logging.info('interop:                    %s',args.interop)
   logging.info('intraop:                    %s',args.intraop)

   config = json.load(open(args.config_filename))
   # config['device'] = device_str
   
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   trainds,testds = data_handler.get_datasets(config)
   test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
   test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

   logger.info('get model')
   net = model.get_model(config)

   loss_func = losses.get_loss(config)

   opt = get_optimizer(config)

   if rank == 0:
      train_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'train')
      test_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'test')

   first_batch = True
   batches_per_epoch = 0
   exit = False
   status_count = config['training']['status']
   batch_size = config['data']['batch_size']
   for epoch_num in range(config['training']['epochs']):
      # Reset the metrics at the start of the next epoch
      test_loss_metric.reset_states()
      test_accuracy_metric.reset_states()

      train_loss_metric = 0
      train_accuracy_metric = 0

      logger.info(f'begin epoch {epoch_num}')

      batch_num = 0
      start = time.time()
      image_rate_sum = 0.
      image_rate_sum2 = 0.
      image_rate_n = 0.
      partial_img_rate = np.zeros(10)
      partial_img_rate_counter = 0
      if rank == args.profrank and args.profiler:
          logger.info('profiling')
          tf.profiler.experimental.start(args.logdir)
      for inputs, labels in trainds:
         loss_value,pred = train_step(net,loss_func,opt,inputs,labels,first_batch,hvd)
         tf.summary.experimental.set_step(batch_num + batches_per_epoch * epoch_num)

         first_batch = False
         batch_num += 1

         train_loss_metric += tf.reduce_mean(loss_value)
         train_accuracy_metric += tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred,-1,tf.int32),tf.cast(labels,tf.int32)),tf.int32)),tf.shape(labels,tf.int32))

         if batch_num % status_count == 0:
            img_per_sec = status_count * batch_size * nranks / (time.time() - start)
            img_per_sec_std = 0
            if batch_num > 1:
                image_rate_n += 1
                image_rate_sum += img_per_sec
                image_rate_sum2 += img_per_sec * img_per_sec
                partial_img_rate[partial_img_rate_counter % 10] = img_per_sec
                partial_img_rate_counter += 1
                img_per_sec = np.mean(partial_img_rate[partial_img_rate>0])
                img_per_sec_std = np.std(partial_img_rate[partial_img_rate>0])
            loss = train_loss_metric / status_count
            acc = (train_accuracy_metric / status_count)[0]
            logger.info(f' [{epoch_num:5d}:{batch_num:5d}]: loss = {loss:10.5f} acc = {acc:10.5f}  imgs/sec = {img_per_sec:7.1f} +/- {img_per_sec_std:7.1f}')
            if rank == 0:
               with train_summary_writer.as_default():
                  step = epoch_num * batches_per_epoch + batch_num
                  tf.summary.experimental.set_step(step)
                  tf.summary.scalar('loss', loss, step=step)
                  tf.summary.scalar('accuracy', acc, step=step)
                  tf.summary.scalar('img_per_sec',img_per_sec,step=step)
                  tf.summary.scalar('learning_rate',opt._decayed_lr(tf.float32))
            start = time.time()
            train_loss_metric = 0
            train_accuracy_metric = 0
            
                
         if args.batch_term == batch_num:
            logger.info('terminating batch training after %s batches',batch_num)
            if rank == args.profrank and args.profiler:
               logger.info('stop profiling')
               tf.profiler.experimental.stop()
            exit = True
            break
      if exit:
         break
      batches_per_epoch = batch_num
      logger.info(f'batches_per_epoch = {batches_per_epoch}')
      
      for test_num,(test_inputs, test_labels) in enumerate(testds):
         test_step(net,loss_func,test_inputs, test_labels,test_loss_metric,test_accuracy_metric)

         if (test_num + 1) % status_count == 0:
            logger.info(f' [{epoch_num:5d}:{test_num:5d}]: test loss = {test_loss_metric.result():10.5f}  test acc = {test_accuracy_metric.result():10.5f}')

      # test_loss = tf.constant(test_loss_metric.result())
      # test_acc = tf.constant(test_accuracy_metric.result())
      # mean_test_loss = hvd.allreduce(test_loss)
      # mean_test_acc = hvd.allreduce(test_acc)

      if rank == 0:
         with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_metric.result(), step=epoch_num * batches_per_epoch + batch_num)
            tf.summary.scalar('accuracy', test_accuracy_metric.result(), step=epoch_num * batches_per_epoch + batch_num)
         ave_img_rate = image_rate_sum / image_rate_n
         std_img_rate = np.sqrt((1/image_rate_n) * image_rate_sum2 - ave_img_rate*ave_img_rate)
         template = 'Epoch {:10.5f}, Loss: {:10.5f}, Accuracy: {:10.5f}, Test Loss: {:10.5f}, Test Accuracy: {:10.5f} Average Image Rate: {:10.5f} +/- {:10.5f}'
         logger.info(template.format(epoch_num + 1,
                               loss,
                               acc * 100,
                               test_loss_metric.result(),
                               test_accuracy_metric.result() * 100,
                               ave_img_rate,
                               std_img_rate))


@tf.function
def train_step(net,loss_func,opt,inputs,labels,first_batch=False,hvd=None,root_rank=0):

   with tf.GradientTape() as tape:
      pred = net(inputs, training=True)
      loss_value = loss_func(labels, tf.cast(pred,tf.float32))
   if hvd:
      tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, net.trainable_variables)
   opt.apply_gradients(zip(grads, net.trainable_variables))
   # Horovod: broadcast initial variable states from rank 0 to all other processes.
   # This is necessary to ensure consistent initialization of all workers when
   # training is started with random weights or restored from a checkpoint.
   #
   # Note: broadcast should be done after the first gradient step to ensure optimizer
   # initialization.
   if hvd and first_batch:
      hvd.broadcast_variables(net.variables, root_rank=root_rank)
      hvd.broadcast_variables(opt.variables(), root_rank=root_rank)

   # tf.print(tf.argmax(tf.nn.softmax(pred,-1),-1),labels)

   return loss_value,pred


@tf.function
def test_step(net,loss_func,inputs,labels,loss_metric,acc_metric):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = net(inputs, training=False)
  loss_value = loss_func(labels, tf.cast(predictions,tf.float32))
  # tf.print(tf.math.reduce_sum(inputs),tf.argmax(tf.nn.softmax(predictions,-1),-1),labels,loss_value)

  loss_metric(loss_value)
  acc_metric(labels, predictions)



def get_optimizer(config):

   # setup learning rate
   lr_schedule = None
   if 'lr_schedule' in config:
      lrs_name = config['lr_schedule']['name']
      lrs_args = config['lr_schedule'].get('args',None)
      if hasattr(tf.keras.optimizers.schedules, lrs_name):
         logger.info(f'using learning rate schedule {lrs_name}')
         lr_schedule = getattr(tf.keras.optimizers.schedules, lrs_name)
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception(f'missing args for learning rate schedule {lrs_name}')

   opt_name = config['optimizer']['name']
   opt_args = config['optimizer'].get('args',None)
   if hasattr(tf.keras.optimizers, opt_name):
      if opt_args:
         if lr_schedule:
            opt_args['learning_rate'] = lr_schedule
         logger.info('passing args to optimizer: %s', opt_args)
         return getattr(tf.keras.optimizers, opt_name)(**opt_args)
      else:
         return getattr(tf.keras.optimizers, opt_name)()
   else:
      raise Exception(f'could not locate optimizer {opt_name}')


if __name__ == "__main__":
   main()
