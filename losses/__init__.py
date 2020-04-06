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
   elif hasattr(tf.keras.losses,loss_name):
      if 'args' in config['loss']:
         logging.info('passing args to loss function: %s',config['loss']['args'])
         return getattr(tf.keras.losses, loss_name)(**config['loss']['args'])
      else:
         return getattr(tf.keras.losses,loss_name)()
   else:
      raise Exception('failed to find loss function %s' % loss_name)
