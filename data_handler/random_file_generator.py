import tensorflow as tf
import logging
import numpy as np
logger = logging.getLogger(__name__)

config = None


def get_datasets(config_file):
   global config
   config = config_file

   train = simple_dataset_from_glob(config['data']['train_glob'])
   valid = simple_dataset_from_glob(config['data']['valid_glob'])

   return train,valid


def simple_dataset_from_glob(glob_string):

   # glob for the input files
   filelist = tf.data.Dataset.list_files(glob_string)
   filelist = filelist.shuffle(10000)
   # parallel_interleave will allow files to be loaded in parallel
   ds = filelist.map(wrapped_load_file_and_preprocess, num_parallel_calls=config['data']['num_parallel_readers'])
   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))

   # shuffle the images and repeat in a performant way
   ds = ds.batch(config['data']['batch_size'],drop_remainder=True)
   # shard the data
   if config['hvd']:
      ds = ds.shard(config['hvd'].size(),config['hvd'].rank())
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   return ds


def wrapped_load_file_and_preprocess(path):
   pyf = tf.py_function(load_file_and_preprocess,[path],(tf.float32,tf.int32))
   return pyf


def load_file_and_preprocess(path):
   logger.info('parsing %s',path)

   img_shape = config['data']['image_shape'] + [config['data']['channels']]
   images = np.float32(np.random.randn(*img_shape))  # HxWxC
   labels = np.random.randint(len(config['data']['classes']))

   # could do some preprocessing here
   logger.info('returning %s, %s',images.shape,labels)
   return (images,labels)  # (HxWxC,1)
