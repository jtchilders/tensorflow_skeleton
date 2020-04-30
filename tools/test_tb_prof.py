import tensorflow as tf
import logging,os
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def get_datasets(batch_size=32):

   mnist = tf.keras.datasets.mnist

   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   x_train = tf.cast(x_train, tf.float32)
   x_test = tf.cast(x_test, tf.float32)
   y_train = tf.cast(y_train, tf.float32)
   y_test = tf.cast(y_test, tf.float32)

   # Add a channels dimension
   x_train = x_train[..., tf.newaxis]
   x_test = x_test[..., tf.newaxis]

   train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(batch_size)

   test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

   return train_ds, test_ds


def main():
   gpus = tf.config.experimental.list_physical_devices('GPU')
   logger.info('gpus = %s',gpus)
   if gpus:
      try:
         for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
         # Visible devices must be set before GPUs have been initialized
         raise

   net = MNIST()

   trainds,testds = get_datasets()

   loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   
   lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(0.001,19,0.5)
   opt = tf.keras.optimizers.Adam(lr_sched)
   train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
   train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

   #tf.profiler.experimental.start('profiler')
   for batch_num,(inputs,labels) in enumerate(trainds):
      with tf.GradientTape() as tape:
         pred = net(inputs, training=True)
         loss_value = loss_func(labels, pred)

      grads = tape.gradient(loss_value, net.trainable_variables)
      opt.apply_gradients(zip(grads, net.trainable_variables))

      train_loss_metric(loss_value)
      train_accuracy_metric(labels, tf.nn.softmax(pred))

      if (batch_num+1) % 20 == 0:
         lr = opt._decayed_lr(tf.float32)
         logger.info(f'[{batch_num}] loss = {train_loss_metric.result():08.5f} acc = {train_accuracy_metric.result():08.5f} lr = {lr:08.5f}')
   #tf.profiler.experimental.stop()

if __name__ == "__main__":
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt)
   main()
