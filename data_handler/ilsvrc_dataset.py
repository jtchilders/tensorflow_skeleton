import tensorflow as tf
import logging,os,glob
import xml.etree.ElementTree as ET
import numpy as np
logger = logging.getLogger(__name__)

labels_hash = None
crop_size = None
def get_datasets(config):
   global labels_hash,crop_size

   base_dir = config['data']['ilsvrc_base_dir']

   crop_size = tf.constant(config['data']['crop_image_size'])

   image_path = os.path.join(base_dir,'Data/CLS-LOC')
   assert os.path.exists(image_path), f'{image_path} not found'
   #annot_path = os.path.join(base_dir,'Annotations/CLS-LOC')
   #assert os.path.exists(annot_path), f'{annot_path} not found'

   labels = glob.glob(os.path.join(image_path,'train') + os.path.sep + '*')
   labels = [ os.path.basename(i) for i in labels ]

   hash_values = tf.range(len(labels))
   hash_keys = tf.constant(labels,dtype=tf.string)

   labels_hash_init = tf.lookup.KeyValueTensorInitializer(hash_keys,hash_values)
   labels_hash = tf.lookup.StaticHashTable(labels_hash_init,-1)

   train_ds = build_dataset(config,image_path,'train')
   valid_ds = build_dataset(config,image_path,'valid')

   return train_ds,valid_ds


def build_dataset(config,image_path,additional_dir):

   dc = config['data']

   image_path = os.path.join(image_path,additional_dir)

   glob_str = image_path + os.path.sep + '*' + os.path.sep + '*'

   # glob for the input files
   filelist = tf.data.Dataset.list_files(glob_str)
   # shuffle and repeat at the input file level
   filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

   # map to read files in parallel
   ds = filelist.map(load_image_and_label, num_parallel_calls=dc['num_parallel_readers'])

   # shard the data
   if config['hvd']:
      ds = ds.shard(config['hvd'].size(), config['hvd'].rank())

   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=dc['prefectch_buffer_size'])

   return ds


def load_image_and_label(image_path):

   label = tf.strings.split(image_path,os.path.sep)[-2]
   # annot_path = tf.strings.regex_replace(image_path,'Data','Annotation')
   # annot_path = tf.strings.regex_replace(annot_path,'\.JPEG','\.xml')

   image = tf.io.read_file(image_path)
   image = tf.image.crop_and_resize(image,tf.constant([0,1]),tf.constant([0]),crop_size)

   return image,labels_hash.lookup(label)


# def parse_xml(filename):
#
#    tree = ET.parse(filename)
#    root = tree.getroot()
#
#    objs = root.findall('object')
#    bndbxs = []
#    for object in objs:
#       bndbox = object.find('bndbox')
