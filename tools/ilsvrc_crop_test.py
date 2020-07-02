import tensorflow as tf
import logging,os,glob
import numpy as np
import xml.etree.ElementTree as ET
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
      numranks = config['hvd'].size()

   filelist = []
   with open(filelist_filename) as file:
      for line in file:
         filelist.append(line.strip())
   batches_per_rank = int(len(filelist) / dc['batch_size'] / numranks)
   logger.info(f'input filelist contains {len(filelist)} files, estimated batches per rank {batches_per_rank}')
   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(filelist)
   # shard the data
   if config['hvd']:
      filelist = filelist.shard(config['hvd'].size(), config['hvd'].rank())
   # shuffle and repeat at the input file level
   logger.debug('starting shuffle')
   filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

   # map to read files in parallel
   logger.debug('starting map')
   # ds = filelist.map(load_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # num_parallel_calls=dc['num_parallel_readers']) #
   ds = filelist.map(load_image_label_bb,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)  # num_parallel_calls=dc['num_parallel_readers']) #
   ds = ds.apply(tf.data.Dataset.unbatch)

   # batch the data
   ds = ds.batch(dc['batch_size']) 

   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # dc['prefectch_buffer_size']) #

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
   logger.debug(f'load_image_and_label %s',image_path)

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
   # tf.print(image_path,label,labels_hash.lookup(label), output_stream=sys.stderr)
   return img,labels_hash.lookup(label)


def load_image_label_bb(image_path):
   logger.debug(f'load_image_and_label %s', image_path)

   label = tf.strings.split(image_path, os.path.sep)[-2]
   annot_path = tf.strings.regex_replace(image_path,'Data','Annotations')
   annot_path = tf.strings.regex_replace(annot_path,'JPEG','xml')

   bounding_boxes,box_indices = tf.py_function(get_bounding_boxes,[annot_path],[tf.float32,tf.int32])

   img = tf.io.read_file(image_path)
   # convert the compressed string to a 3D uint8 tensor
   img = tf.image.decode_jpeg(img, channels=3)
   img = tf.expand_dims(img,0)

   # create individual images based on bounding boxes
   imgs = tf.image.crop_and_resize(img,bounding_boxes,box_indices,crop_size)

   # Use `convert_image_dtype` to convert to floats in the [0,1] range.
   imgs = tf.image.convert_image_dtype(imgs, tf.float16)
   # resize the image to the desired size.
   imgs = tf.image.resize(imgs, crop_size)
   # tf.print(image_path,label,labels_hash.lookup(label), output_stream=sys.stderr)
   label = labels_hash.lookup(label)
   labels = tf.fill([tf.shape(imgs)[0]],label)
   return imgs, labels


def get_bounding_boxes(filename):
   filename = bytes.decode(filename.numpy())
   logger.debug('xml file: %s',filename)
   try:
      tree = ET.parse(filename)
      root = tree.getroot()

      img_size = root.find('size')
      img_width = int(img_size.find('width').text)
      img_height = int(img_size.find('height').text)
      # img_depth = int(img_size.find('depth').text)

      objs = root.findall('object')
      bndbxs = []
      # label = None
      for object in objs:
         # label = object.find('name').text
         bndbox = object.find('bndbox')
         bndbxs.append([
            float(bndbox.find('ymin').text) / (img_height - 1),
            float(bndbox.find('xmin').text) / (img_width - 1),
            float(bndbox.find('ymax').text) / (img_height - 1),
            float(bndbox.find('xmax').text) / (img_width - 1)
         ])
   except FileNotFoundError:
      bndbxs = [[0,0,1,1]]

   return np.asarray(bndbxs,np.float),np.zeros(len(bndbxs))



if __name__ == '__main__':
   logging.basicConfig(level=logging.DEBUG)
   import argparse,json
   parser = argparse.ArgumentParser(description='test this')
   parser.add_argument('-c', '--config', dest='config_filename',
                       help='configuration filename in json format',
                       required=True)
   args = parser.parse_args()

   summary_writer = tf.summary.create_file_writer('ilsvrc_tblog')

   config = json.load(open(args.config_filename))

   crop_size = tf.constant(config['data']['crop_image_size'])
   filelist_filename = config['data']['train_filelist']

   filelist = []
   j = 0
   with open(filelist_filename) as file:
      for i,image_filename in enumerate(file):
         image_filename = image_filename.strip()
         logger.info('i = %s; image fn = %s',i,image_filename)
         xml_filename = image_filename.replace('Data','Annotations').replace('.JPEG','.xml')

         bb,bi = get_bounding_boxes(tf.convert_to_tensor(xml_filename))
         if bb.shape[0] == 1: continue

         logger.info('bb = %s; bi= %s',bb,bi)

         assert os.path.exists(image_filename), f'"{image_filename}" not found'
         img = tf.io.read_file(image_filename)
         # convert the compressed string to a 3D uint8 tensor
         img = tf.image.decode_jpeg(img, channels=3)
         img = tf.expand_dims(img, 0)
         logger.info(f'image shape = {img.shape}  type = {type(img)}  dtype = {img.dtype}')
         logger.info(f'image= {img}')
         with summary_writer.as_default():
            tf.summary.image("raw images", img,step=j)

         # create individual images based on bounding boxes
         imgs = tf.image.crop_and_resize(img, bb, bi, crop_size)
         imgs = tf.cast(imgs,tf.uint8)
         # colors = [[255.,0.,0.]]
         # imgs = tf.image.draw_bounding_boxes(tf.cast(img,tf.float32),tf.cast(tf.expand_dims(bb,0),tf.float32),tf.convert_to_tensor(colors,tf.float32))
         # imgs = tf.cast(imgs,tf.uint8)
         # logger.info(f'images shape = {imgs.shape}  type = {type(imgs)} dtype = {imgs.dtype}')
         # logger.info(f'image= {imgs}')
         with summary_writer.as_default():
            tf.summary.image("cropped images", imgs,step=j)
         j += 1
         if j >= 10: break


