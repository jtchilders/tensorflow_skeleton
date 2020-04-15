import tensorflow as tf
import logging,os,glob
import xml.etree.ElementTree as ET
import numpy as np
logger = logging.getLogger(__name__)

labels_hash = None
crop_size = None
def get_datasets(config):
   global labels_hash,crop_size
   logger.debug('get dataset')

   crop_size = tf.constant(config['data']['crop_image_size'])
   train_filelist = config['data']['train_filelist']
   test_filelist = config['data']['test_filelist']

   assert os.path.exists(train_filelist), f'{train_filelist} not found'
   assert os.path.exists(test_filelist), f'{test_filelist} not found'


   ## Create a hash table for labels from string to int
   with open(train_filelist) as file:
      filepath = file.readline().strip()

   label_path = '/'.join(filepath.split('/')[:-2])
   labels = glob.glob(label_path + os.path.sep + '*')
   logger.debug(f'num labels: {len(labels)}')
   labels = [os.path.basename(i) for i in labels]
   hash_values = tf.range(len(labels))
   hash_keys = tf.constant(labels, dtype=tf.string)
   labels_hash_init = tf.lookup.KeyValueTensorInitializer(hash_keys, hash_values)
   labels_hash = tf.lookup.StaticHashTable(labels_hash_init, -1)

   train_ds = build_dataset_from_filelist(config,train_filelist)
   valid_ds = build_dataset_from_filelist(config,test_filelist)

   return train_ds,valid_ds


def build_dataset_from_filelist(config,filelist_filename):
   logger.debug(f'build dataset {filelist_filename}')

   dc = config['data']

   numranks = 1
   if config['hvd']:
      numranks = config['hvd'].Get_size()

   filelist = []
   with open(filelist_filename) as file:
      for line in file:
         filelist.append(line.strip())
   batches_per_rank = len(filelist) / dc['batch_size'] / numranks
   logger.info(f'input filelist contains {len(filelist)} files, estimated f{batches_per_rank}')
   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(filelist)
   # shuffle and repeat at the input file level
   logger.debug('starting shuffle')
   filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

   # map to read files in parallel
   logger.debug('starting map')
   ds = filelist.map(load_image_and_label, num_parallel_calls=dc['num_parallel_readers'])

   # batch the data
   ds = ds.batch(dc['batch_size'])

   # shard the data
   if config['hvd']:
      ds = ds.shard(config['hvd'].size(), config['hvd'].rank())

   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=dc['prefectch_buffer_size'])

   return ds


def build_dataset_from_glob(config,image_path,glob_str):
   logger.debug(f'build dataset {glob_str}')

   dc = config['data']

   glob_str = os.path.join(image_path,glob_str)

   # glob for the input files
   filelist = tf.data.Dataset.list_files(glob_str)
   # shuffle and repeat at the input file level
   logger.debug(f'starting shuffle')
   filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

   # map to read files in parallel
   logger.debug(f'starting map')
   ds = filelist.map(load_image_and_label, num_parallel_calls=dc['num_parallel_readers'])

   # batch the data
   ds = ds.batch(dc['batch_size'])

   # shard the data
   if config['hvd']:
      ds = ds.shard(config['hvd'].size(), config['hvd'].rank())

   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=dc['prefectch_buffer_size'])

   return ds


def load_image_and_label(image_path):
   logger.debug(f'load_image_and_label {image_path}')

   label = tf.strings.split(image_path,os.path.sep)[-2]
   # annot_path = tf.strings.regex_replace(image_path,'Data','Annotation')
   # annot_path = tf.strings.regex_replace(annot_path,'\.JPEG','\.xml')

   img = tf.io.read_file(image_path)
   # convert the compressed string to a 3D uint8 tensor
   img = tf.image.decode_jpeg(img, channels=3)
   # Use `convert_image_dtype` to convert to floats in the [0,1] range.
   img = tf.image.convert_image_dtype(img, tf.float16)
   # resize the image to the desired size.
   img = tf.image.resize(img, crop_size)

   return img,labels_hash.lookup(label)


# def parse_xml(filename):
#
#    tree = ET.parse(filename)
#    root = tree.getroot()
#
#    objs = root.findall('object')
#    bndbxs = []
#    for object in objs:
#       bndbox = object.find('bndbox')
