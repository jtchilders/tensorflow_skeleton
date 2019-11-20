import tensorflow as tf
import logging
logger = logging.getLogger('loss')

__all__ = ['focal_loss_softmax']
from . import focal_loss_softmax


def get_loss(config):
   loss_name = config['loss']['name']
   if loss_name in globals():
      logger.info('using loss name %s',loss_name)
      return globals()[loss_name]
   elif hasattr(tf.nn,loss_name):
      return getattr(tf.nn,loss_name)
   else:
      raise Exception('failed to find data handler %s in globals %s' % (loss_name,globals()))
