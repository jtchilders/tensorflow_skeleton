import tensorflow as tf,logging
from . import tf_layer_helper as tlh
logger = logging.getLogger(__name__)


def get_model(input_pl,is_training_pl,config):

   net = tlh.conv2d(input_pl,64,(3,3),'conv1',is_training=is_training_pl)
   net = tlh.conv2d(net,128,(3,3),'conv2',is_training=is_training_pl)
   net = tlh.conv2d(net,64,(3,3),'conv3',is_training=is_training_pl)

   net = tf.reshape(net,[net.get_shape()[0].value,-1])

   net = tlh.fully_connected(net,128,'fc1',is_training=is_training_pl)
   net = tlh.fully_connected(net,len(config['data']['classes']),'fc2',is_training=is_training_pl)
   logger.info('net = %s',net)
   return net
