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
   # parallel_interleave will allow files to be loaded in parallel
   ds = filelist.apply(tf.contrib.data.parallel_interleave(load_file_and_preprocess, cycle_length=config['data']['num_parallel_readers']))
   # shuffle the images and repeat in a performant way
   ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   ds = ds.batch(config['data']['batch_size'])
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   return ds


def load_file_and_preprocess(path):
   logger.info('parsing %s',path)

   img_shape = [config['data']['imgs_per_file']] + config['data']['image_shape']
   images = np.random.randn(*img_shape)
   labels = np.random.randint(config['data']['imgs_per_file'])

   # could do some preprocessing here
   #logger.info('returning %s, %s',images,labels)
   return tf.data.Dataset.from_tensors((images,labels))
