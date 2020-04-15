#!/usr/bin/env python
import argparse,logging,json,time,os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import device_lib
import data_handler
import model,lr_func,losses,accuracies

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()
DEFAULT_LOGDIR = '/tmp/tf-' + str(os.getpid())


def main():
   ''' simple starter program for tensorflow models. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',type=int,help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',type=int,help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('-l','--logdir',default=DEFAULT_LOGDIR,help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('--horovod', default=False, action='store_true', help="Use MPI with horovod")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   hvd = None
   rank = 0
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
      if rank > 0:
         logging_level = logging.WARNING

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

   device_str = '/CPU:0'
   if tf.test.is_gpu_available():
      # if using MPI, assume one rank per GPU
      if hvd:
         logger.warning('Setting CUDA_VISIBLE_DEVICES to be the GPU for this local rank, ASSUMES 1 GPU per MPI RANK')
         os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
      gpus = tf.config.experimental.list_logical_devices('GPU')
      logger.warning('gpus = %s', gpus)
      device_str = gpus[0].name

   logger.warning('device:                     %s', device_str)

   if 'CUDA_VISIBLE_DEVICES' in os.environ:
      logging.warning('CUDA_VISIBLE_DEVICES=%s %s',os.environ['CUDA_VISIBLE_DEVICES'],device_lib.list_local_devices())
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
   config['device'] = device_str
   
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   tf.config.threading.set_inter_op_parallelism_threads(args.interop)
   tf.config.threading.set_intra_op_parallelism_threads(args.intraop)

   trainds,testds = data_handler.get_datasets(config)

   logger.info('get model')
   net = model.get_model(config)

   loss_func = losses.get_loss(config)

   opt = get_optimizer(config)

   train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
   train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

   test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
   test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

   if rank == 0:
      train_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'train')
      test_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'test')

   first_batch = True
   batches_per_epoch = 0
   for epoch_num in range(config['training']['epochs']):
      # Reset the metrics at the start of the next epoch
      train_loss_metric.reset_states()
      train_accuracy_metric.reset_states()
      test_loss_metric.reset_states()
      test_accuracy_metric.reset_states()

      logger.info(f'begin epoch {epoch_num}')

      batch_num = 0
      start = time.time()
      for inputs, labels in trainds:
         train_step(net,loss_func,opt,inputs,labels,train_loss_metric,train_accuracy_metric,first_batch,hvd)

         first_batch = False
         batch_num += 1
         if rank == 0 and batch_num % config['training']['status'] == 0:
            img_per_sec = config['training']['status'] * config['data']['batch_size'] / (time.time() - start)
            logger.info(f' [{epoch_num:5d}:{batch_num:5d}]: loss = {train_loss_metric.result():10.5f} acc = {train_accuracy_metric.result():10.5f}  imgs/sec = {img_per_sec}')
            logger.info('%s',opt.__dict__)

            with train_summary_writer.as_default():
               tf.summary.scalar('loss', train_loss_metric.result(), step=epoch_num * batches_per_epoch + batch_num)
               tf.summary.scalar('accuracy', train_accuracy_metric.result(), step=epoch_num * batches_per_epoch + batch_num)
            start = time.time()

      batches_per_epoch = batch_num
      logger.info(f'batches_per_epoch = {batches_per_epoch}')

      for test_inputs, test_labels in testds:
         test_step(net,loss_func,test_inputs, test_labels,test_loss_metric,test_accuracy_metric)

      test_loss = tf.constant(test_loss_metric.result())
      test_acc = tf.constant(test_accuracy_metric.result())
      mean_test_loss = hvd.allreduce(test_loss)
      mean_test_acc = hvd.allreduce(test_acc)

      if rank == 0:
         with test_summary_writer.as_default():
            tf.summary.scalar('loss', mean_test_loss, step=epoch_num * batches_per_epoch + batch_num)
            tf.summary.scalar('accuracy', mean_test_acc, step=epoch_num * batches_per_epoch + batch_num)

         template = 'Epoch {:10.5f}, Loss: {:10.5f}, Accuracy: {:10.5f}, Test Loss: {:10.5f}, Test Accuracy: {:10.5f}'
         logger.info(template.format(epoch_num + 1,
                               train_loss_metric.result(),
                               train_accuracy_metric.result() * 100,
                               test_loss_metric.result(),
                               test_accuracy_metric.result() * 100))


@tf.function
def train_step(net,loss_func,opt,inputs,labels,loss_metric,acc_metric,first_batch=False,hvd=None,root_rank=0):

   with tf.GradientTape() as tape:
      pred = net(inputs, training=True)
      loss_value = loss_func(labels, pred)
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
   loss_metric(loss_value)
   acc_metric(labels,pred)

   # return loss_value,pred

@tf.function
def test_step(net,loss_func,inputs,labels,loss_metric,acc_metric):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = net(inputs, training=False)
  t_loss = loss_func(labels, predictions)

  loss_metric(t_loss)
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