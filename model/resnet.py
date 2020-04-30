import tensorflow as tf


def get_model(config):
   net = ResNet50(config)
   return net

class ResNet50(tf.keras.Model):
   def __init__(self,config):
      super(ResNet50,self).__init__(name='')

      self.conv2a = tf.keras.layers.Conv2D(64, (7, 7),strides=(2,2),padding='same',name='conv_input')
      self.bn2a = tf.keras.layers.BatchNormalization(name='bn_input')

      self.poolA = tf.keras.layers.MaxPool2D((3,3),(2,2))

      self.cbA = ResnetConvBlock(3,[64,64,256],stage=2,block='a',strides=(1,1))
      self.idAa = ResnetIdentityBlock(3,[64,64,256],stage=2,block='b')
      self.idAb = ResnetIdentityBlock(3,[64,64,256],stage=2,block='c')

      self.cbB = ResnetConvBlock(3,[128, 128, 512],stage=3,block='a')
      self.idBa = ResnetIdentityBlock(3,[128, 128, 512],stage=3,block='b')
      self.idBb = ResnetIdentityBlock(3,[128, 128, 512],stage=3,block='c')
      self.idBc = ResnetIdentityBlock(3,[128, 128, 512],stage=3,block='d')

      self.cbC = ResnetConvBlock(3,[256, 256, 1024],stage=4,block='a')
      self.idCa = ResnetIdentityBlock(3,[256, 256, 1024],stage=4,block='b')
      self.idCb = ResnetIdentityBlock(3,[256, 256, 1024],stage=4,block='c')
      self.idCc = ResnetIdentityBlock(3,[256, 256, 1024],stage=4,block='d')
      self.idCd = ResnetIdentityBlock(3,[256, 256, 1024],stage=4,block='e')
      self.idCe = ResnetIdentityBlock(3,[256, 256, 1024],stage=4,block='f')

      self.cbD = ResnetConvBlock(3,[512, 512, 2048],stage=5,block='a')
      self.idDa = ResnetIdentityBlock(3,[512, 512, 2048],stage=5,block='b')
      self.idDb = ResnetIdentityBlock(3,[512, 512, 2048],stage=5,block='c')

      self.poolB = tf.keras.layers.AveragePooling2D((7,7),strides=(7,7))

      self.flatten = tf.keras.layers.Flatten()

      self.dense = tf.keras.layers.Dense(config['data']['num_classes'],name='fc%s' % config['data']['num_classes'])

   def call(self, input_tensor, training=False):
      x = self.conv2a(input_tensor)
      x = self.bn2a(x, training=training)
      x = tf.nn.relu(x)

      x = self.poolA(x)

      x = self.cbA(x,training)
      x = self.idAa(x,training)
      x = self.idAb(x,training)

      x = self.cbB(x,training)
      x = self.idBa(x,training)
      x = self.idBb(x,training)
      x = self.idBc(x,training)

      x = self.cbC(x,training)
      x = self.idCa(x,training)
      x = self.idCb(x,training)
      x = self.idCc(x,training)
      x = self.idCd(x,training)
      x = self.idCe(x,training)

      x = self.cbD(x,training)
      x = self.idDa(x,training)
      x = self.idDb(x,training)

      x = self.poolB(x)

      x = self.flatten(x)

      return self.dense(x)


class ResnetIdentityBlock(tf.keras.Model):
   def __init__(self, kernel_size, filters, stage, block,data_format='channels_last'):
      super(ResnetIdentityBlock, self).__init__(name='')
      filters1, filters2, filters3 = filters

      conv_name_base = 'res' + str(stage) + block + '_branch'
      bn_name_base = 'bn' + str(stage) + block + '_branch'
      bn_axis = 1 if data_format == 'channels_first' else 3

      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1),name=conv_name_base + '_2a', data_format=data_format)
      self.bn2a = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2a')

      self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '_2b', data_format=data_format)
      self.bn2b = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2b')

      self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1),name=conv_name_base + '_2c', data_format=data_format)
      self.bn2c = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2c')

   def call(self, input_tensor, training=False):
      x = self.conv2a(input_tensor)
      x = self.bn2a(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2b(x)
      x = self.bn2b(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2c(x)
      x = self.bn2c(x, training=training)

      x += input_tensor
      return tf.nn.relu(x)


class ResnetConvBlock(tf.keras.Model):
   def __init__(self, kernel_size, filters, stage, block, data_format='channels_last', strides=(2, 2)):
      super(ResnetConvBlock, self).__init__(name='')
      filters1, filters2, filters3 = filters

      conv_name_base = 'res' + str(stage) + block + '_branch'
      bn_name_base = 'bn' + str(stage) + block + '_branch'
      bn_axis = 1 if data_format == 'channels_first' else 3

      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1),strides=strides,name=conv_name_base + '_2a', data_format=data_format)
      self.bn2a = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2a')

      self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '_2b', data_format=data_format)
      self.bn2b = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2b')

      self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1),name=conv_name_base + '_2c', data_format=data_format)
      self.bn2c = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2c')

      self.conv2d = tf.keras.layers.Conv2D(filters3, (1, 1),strides=strides,name=conv_name_base + '_2d', data_format=data_format)
      self.bn2d = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_2d')

   def call(self, input_tensor, training=False):
      x = self.conv2a(input_tensor)
      x = self.bn2a(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2b(x)
      x = self.bn2b(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2c(x)
      x = self.bn2c(x, training=training)

      shortcut = self.conv2d(input_tensor)
      shortcut = self.bn2d(shortcut)

      x += shortcut

      return tf.nn.relu(x)
