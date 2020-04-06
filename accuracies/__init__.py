import logging
import tensorflow as tf
logger = logging.getLogger('accuracies')

__all__ = ['iou_accuracy','simple_class_accuracy']
from . import iou_accuracy,simple_class_accuracy


def get_accuracy(config):
   acc_name = config['accuracy']['name']
   if acc_name in globals():
      logger.info('using accuracy name %s',acc_name)
      return globals()[acc_name].get_accuracy(config)
   elif hasattr(tf.keras.metrics,acc_name):
      return getattr(tf.keras.metrics,acc_name)
   else:
      raise Exception('failed to find accuracy %s in globals %s' % (acc_name,globals()))
