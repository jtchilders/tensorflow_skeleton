import tensorflow as tf,logging
logger = logging.getLogger(__name__)


def get_model(config):
   net = MNIST()
   return net


class MNIST(tf.keras.Model):

   def __init__(self):
      super(MNIST, self).__init__()
      self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
      self.flatten = tf.keras.layers.Flatten()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(10)

   def call(self, x):
      x = self.conv1(x)
      x = self.flatten(x)
      x = self.d1(x)
      return self.d2(x)

