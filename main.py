#!/usr/bin/env python
import argparse,logging,json,time,os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import device_lib
import data_handler
import model,lr_func,losses,accuracies
import horovod
import horovod.tensorflow as hvd
hvd.init()
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()
DEFAULT_LOGDIR = '/tmp/tf-' + str(os.getpid())


def main():
   ''' simple starter program for tensorflow models. '''
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + ('%05d' % hvd.rank()) + ':%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('-l','--logdir',default=DEFAULT_LOGDIR,help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

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
   
   logging.warning('rank: %5d   size: %5d  local rank: %5d  local size: %5d',hvd.rank(),hvd.size(),hvd.local_rank(),hvd.local_size())
   if 'CUDA_VISIBLE_DEVICES' in os.environ:
      logging.warning('CUDA_VISIBLE_DEVICES=%s %s',os.environ['CUDA_VISIBLE_DEVICES'],device_lib.list_local_devices())
   else:
      logging.info('CUDA_VISIBLE_DEVICES not defined in os.environ')
   logging.info('using tensorflow version:   %s',tf.__version__)
   logging.info('using tensorflow from:      %s',tf.__file__)
   logging.info('using horovod version:      %s',horovod.__version__)
   logging.info('using horovod from:         %s',horovod.__file__)
   logging.info('logdir:                     %s',args.logdir)
   logging.info('interop:                    %s',args.interop)
   logging.info('intraop:                    %s',args.intraop)
   
   device_str = '/CPU:0'
   if tf.test.is_gpu_available():
      # device_str = '/device:GPU:' + str(hvd.local_rank())
      gpus = tf.config.experimental.list_logical_devices('GPU')
      logger.warning('gpus = %s',gpus)
      # assert hvd.local_rank() < len(gpus), f'localrank = {hvd.local_rank()} len(gpus) = {len(gpus)}'
      device_str = gpus[0].name
      # logger.info('device_str = %s',device_str)
      
   logger.warning('device:                     %s',device_str)

   config = json.load(open(args.config_filename))
   config['device'] = device_str
   
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   with tf.Graph().as_default():
      logger.info('getting datasets')
      trainds,validds = data_handler.get_datasets(config)

      input_shape = (config['data']['batch_size'],config['data']['image_shape'][0],config['data']['image_shape'][1],config['data']['channels'])
      target_shape = (config['data']['batch_size'])  #,len(config['data']['classes']))
      iterator = tf.compat.v1.data.Iterator.from_structure((tf.float32,tf.int32),(input_shape,target_shape))
      input,target = iterator.get_next()
      training_init_op = iterator.make_initializer(trainds)
      valid_init_op = iterator.make_initializer(validds)

      with tf.device(device_str):

         is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
         batch = tf.Variable(0)
         batch_size = tf.constant(config['data']['batch_size'])

         pred = model.get_model(input,is_training_pl,config)
         logger.info('pred = %s  target = %s',pred,target)
         # pred = BxC, target = BxC
         loss = losses.get_loss(config)(labels=target,logits=pred)
         tf.compat.v1.summary.scalar('loss/combined',loss)

         accuracy = accuracies.get_accuracy(pred,target,config)
         tf.compat.v1.summary.scalar('accuracy/combined', accuracy)

         #learning_rate = pointnet_seg.get_learning_rate(batch,config) * hvd.size()
         learning_rate = lr_func.get_learning_rate(batch * batch_size,config)
         tf.compat.v1.summary.scalar('learning_rate',learning_rate)
         if config['optimizer']['name'] == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

         # adding Horovod distributed optimizer
         optimizer = hvd.DistributedOptimizer(optimizer)

         # create the training operator
         train_op = optimizer.minimize(loss, global_step=batch)

         # Add ops to save and restore all the variables.
         saver = tf.train.Saver()

         merged = tf.compat.v1.summary.merge_all()

      logger.info('create session')

      config_proto = tf.compat.v1.ConfigProto()
      if 'gpu' in device_str:
         config_proto.gpu_options.allow_growth = True
         config_proto.gpu_options.visible_device_list = os.environ['CUDA_VISIBLE_DEVICES']
      else:
         config_proto.allow_soft_placement = True
         config_proto.intra_op_parallelism_threads = args.intraop
         config_proto.inter_op_parallelism_threads = args.interop

      # Initialize an iterator over a dataset with 10 elements.
      sess = tf.compat.v1.Session(config=config_proto)
      
      # create tensorboard writers
      if hvd.rank() == 0:
         train_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.logdir, 'train'),sess.graph)
         valid_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.logdir, 'valid'),sess.graph)
      
      # initialize global vars and horovod broadcast initial model
      init = tf.compat.v1.global_variables_initializer()
      sess.run(init, {is_training_pl: True})
      sess.run(hvd.broadcast_global_variables(0))
      
      logger.info('running over data')
      status_interval = config['training']['status']
      total_acc = 0.
      loss_sum = 0.
      for epoch in range(config['training']['epochs']):
         logger.info('epoch %s of %s',epoch + 1,config['training']['epochs'])

         # initialize the data iterator for training loop
         sess.run(training_init_op)
         
         # training loop
         start = time.time()
         while True:
            try:
               # set that we are training
               feed_dict = {is_training_pl: True}
               summary, step, _, loss_val, accuracy_val = sess.run([merged, batch,
                   train_op, loss, accuracy], feed_dict=feed_dict)
               
               # only one rank should write summary or file gets clobbered
               if hvd.rank() == 0:
                  train_writer.add_summary(summary, step)
               
               # keep running average of the accuracy and loss for status print
               total_acc += accuracy_val
               loss_sum += loss_val
               
               # report status periodically
               if step % status_interval == 0:
                  end = time.time()
                  duration = end - start
                  logger.info('step: %10d   mean loss: %10.6f   accuracy:  %10.6f  imgs/sec: %10.6f',
                              step,
                              loss_sum / float(status_interval),total_acc / float(status_interval),
                              float(status_interval) * config['data']['batch_size'] / duration)
                  start = time.time()
                  total_acc = 0.
                  loss_sum = 0.
            # exception thrown when data is done
            except tf.errors.OutOfRangeError:
               logger.info(' end of epoch ')
               saver.save(sess, os.path.join(args.logdir, "model.ckpt"),global_step=step)
               break

         logger.info('running validation')
         # initialize the validation data iterator
         sess.run(valid_init_op)
         total_acc = 0.
         total_loss = 0.
         steps = 0.
         while True:
            try:
               # tell model we are not training
               feed_dict = {is_training_pl: False}
               # only need to calc loss and accuracy this time
               summary, step, loss_val, accuracy_val = sess.run([merged, batch, loss, accuracy], feed_dict=feed_dict)
               total_acc += accuracy_val
               total_loss += loss_val
               steps += 1.

               # only rank 0 writes the summary
               # note: in the current implementation, 'step' doesn't change
               # resulting in every data point going into the same step
               # not sure of a better way to do this.
               if hvd.rank() == 0:
                  valid_writer.add_summary(summary, step)
            except tf.errors.OutOfRangeError:
               # report results
               total_loss = total_loss / steps
               total_acc = total_acc / steps
               logger.info(' end of validation  mean loss: %10.6f  mean acc: %10.6f',total_loss,total_acc)
               break
         



if __name__ == "__main__":
   main()
