import tensorflow as tf
import logging,glob
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

config = None


def get_datasets(config_file):
   global config
   config = config_file

   if 'train_glob' in config['data'] and 'valid_glob' in config['data']:
      train = simple_dataset_from_glob(config['data']['train_glob'])
      valid = simple_dataset_from_glob(config['data']['valid_glob'])
   elif 'glob' in config_file['data'] and 'train_fraction' in config_file['data']:
      train,valid = split_filelists(config['data']['glob'],config['data']['train_fraction'])
      train = simple_dataset_from_filelist(train)
      valid = simple_dataset_from_filelist(valid)
   else:
      raise Exception('data set reading not configured correctly')

   return train,valid


def simple_dataset_from_glob(glob_string):

   # glob for the input files
   filelist = tf.data.Dataset.list_files(glob_string)
   # shuffle and repeat at the input file level
   filelist = filelist.shuffle(10000)
   # filelist = filelist.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   # map to read files in parallel
   ds = filelist.map(load_file_and_preprocess,num_parallel_calls=config['data']['num_parallel_readers'])
   
   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
   
   # speficy batch size
   ds = ds.batch(config['data']['batch_size'],drop_remainder=True)

   # shard the data
   ds = ds.shard(config['hvd'].size(),config['hvd'].rank())
   
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   
   return ds


def simple_dataset_from_filelist(filelist):

   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(np.array(filelist))
   # shuffle and repeat at the input file level
   filelist = filelist.shuffle(10000)
   # filelist = filelist.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   # map to read files in parallel
   ds = filelist.map(load_file_and_preprocess,num_parallel_calls=config['data']['num_parallel_readers'])
   
   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
   
   # speficy batch size
   ds = ds.batch(config['data']['batch_size'],drop_remainder=True)

   # shard the data
   ds = ds.shard(config['hvd'].size(),config['hvd'].rank())
   
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   
   return ds


def split_filelists(glob_str,train_fraction):

   logger.info('using glob string to find input files: %s',glob_str)

   # glob for full filelist
   full_filelist = sorted(glob.glob(glob_str))
   nfiles = len(full_filelist)

   ntrain = int(nfiles * train_fraction)

   train_filelist = full_filelist[:ntrain]
   valid_filelist = full_filelist[ntrain:]

   logger.info('number train: %s  number valid: %s',len(train_filelist),len(valid_filelist))

   return train_filelist,valid_filelist


def load_file_and_preprocess(path):
   pyf = tf.py_function(wrapped_loader,[path],(tf.float32,tf.int32))
   return pyf


def wrapped_loader(path):
   path = path.numpy().decode('utf-8')
   col_names    = ['type','data','time','etc']
   col_dtype    = {'type': np.int64, 'data': np.float32, 'time': np.float32, 'etc': np.float32}

   data = pd.read_csv(path,header=None,names=col_names, dtype=col_dtype, sep='\t')
   
   # could do some preprocessing here
   return (get_input(data),get_labels(data))


def get_input(data):
   input = data[['data','time','etc']].to_numpy()

   # apply normalization
   min = input.min(0)
   max = input.max(0)
   input = (input - min) / (max - min)
   input = np.float32(input)

   # pad vectors if needed
   img_shape = (1,config['data']['num_points'],config['data']['num_features'])
   padded_input = np.zeros(img_shape,dtype=np.float32)

   padded_input[0,:input.shape[0],:] = input

   # input = padded_input.transpose()
   
   return padded_input


def get_labels(data):
   target = data['type']
   # can map inputs to a linear class index if needed
   # map_pid = {-99:0,-11:2,11:2,0:1}
   # target = target.map(map_pid)
   target = np.int32(target.to_numpy())

   # pad vectors if needed
   padded_target = np.zeros((1,config['data']['num_points']),dtype=np.int32)
   padded_target[0,:target.shape[0]] = target

   return padded_target

