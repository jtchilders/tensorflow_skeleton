#!/usr/bin/env python
import argparse,logging,json,time,os
import tensorflow as tf
import data_handler
import horovod
import horovod.tensorflow as hvd
hvd.init()
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()


def main():
   ''' simple starter program for tensorflow models. '''
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + ('%05d' % hvd.rank()) + ':%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logging.info('using tensorflow version:   %s',tf.__version__)
   logging.info('using tensorflow from:      %s',tf.__file__)
   logging.info('using horovod version:      %s',horovod.__version__)
   logging.info('using horovod from:         %s',horovod.__file__)
   
   config = json.load(open(args.config_filename))

   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))

   logger.info('getting datasets')
   trainds,validds = data_handler.get_datasets(config)

   logger.info('got datasets')

   train_itr = trainds.make_initializable_iterator()
   #train_itr = trainds.make_one_shot_iterator()
   next_train = train_itr.get_next()

   logger.info('create session')

   config = tf.ConfigProto()
   config.allow_soft_placement = True
   config.intra_op_parallelism_threads = args.intraop
   config.inter_op_parallelism_threads = args.interop

   # Initialize an iterator over a dataset with 10 elements.
   sess = tf.Session(config=config)
   logger.info('initialize dataset iterator')
   sess.run(train_itr.initializer)
   logger.info('running over data')
   for i in range(config['training']['epochs']):
      logger.info('epoch %s of %s',i+1,config['training']['epochs'])
      data_start = time.time()
      value = sess.run(next_train)
      data_end = time.time()

      logger.info('data time = %s value = %s %s',data_end - data_start,value[0].shape,value[1].shape)



if __name__ == "__main__":
   main()
