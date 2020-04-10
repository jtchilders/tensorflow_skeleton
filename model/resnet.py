import tensorflow as tf


class ResNet50(tf.keras.Model):
   def __init__(self):
      super(ResNet50,self).__init__(name='')

      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
      self.bn2a = tf.keras.layers.BatchNormalization()

      self.poolA = tf.keras.layers.MaxPool2D((3,3),(2,2))

      self.cbA = ResnetConvBlock(3,[64,64,256])
      self.idAa = ResnetIdentityBlock(3,[64,64,256])
      self.idAb = ResnetIdentityBlock(3,[64,64,256])

      self.cbB = ResnetConvBlock(3,[128, 128, 512])
      self.idBa = ResnetIdentityBlock(3,[128, 128, 512])
      self.idBb = ResnetIdentityBlock(3,[128, 128, 512])
      self.idBc = ResnetIdentityBlock(3,[128, 128, 512])

      self.cbC = ResnetConvBlock(3,[256, 256, 1024])
      self.idCa = ResnetIdentityBlock(3,[256, 256, 1024])
      self.idCb = ResnetIdentityBlock(3,[256, 256, 1024])
      self.idCc = ResnetIdentityBlock(3,[256, 256, 1024])
      self.idCd = ResnetIdentityBlock(3,[256, 256, 1024])
      self.idCe = ResnetIdentityBlock(3,[256, 256, 1024])

      self.cbD = ResnetConvBlock(3,[512, 512, 2048])
      self.idDa = ResnetIdentityBlock(3,[512, 512, 2048])
      self.idDb = ResnetIdentityBlock(3,[512, 512, 2048])

      self.poolB = tf.keras.layers.AveragePooling2D((2,2))

      self.dense = tf.keras.layers.Dense()

   def call(self, input_tensor, training=False):
      x = tf.nn.relu(self.bn2a(self.conv2a(input_tensor)))

      x = self.poolA(x)

      x = self.idAb(self.idAa(self.cbA(x,training),training),training)

      x = self.idBc(self.idBb(self.idBa(self.cbB(x,training),training),training),training)

      x = self.idCe(self.idCd(self.idCc(self.idCb(self.idCa(self.cbC(x,training),training),training),training),training),training)

      x = self.idDb(self.idDa(self.cbD(x,training),training),training)

      x = self.poolB(x)

      return self.dense(x)


class ResnetIdentityBlock(tf.keras.Model):
   def __init__(self, kernel_size, filters):
      super(ResnetIdentityBlock, self).__init__(name='')
      filters1, filters2, filters3 = filters

      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
      self.bn2a = tf.keras.layers.BatchNormalization()

      self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
      self.bn2b = tf.keras.layers.BatchNormalization()

      self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
      self.bn2c = tf.keras.layers.BatchNormalization()

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
   def __init__(self, kernel_size, filters):
      super(ResnetConvBlock, self).__init__(name='')
      filters1, filters2, filters3 = filters

      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
      self.bn2a = tf.keras.layers.BatchNormalization()

      self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
      self.bn2b = tf.keras.layers.BatchNormalization()

      self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
      self.bn2c = tf.keras.layers.BatchNormalization()

      self.conv2d = tf.keras.layers.Conv2D(filters3, (1, 1))
      self.bn2d = tf.keras.layers.BatchNormalization()

   def call(self, input_tensor, training=False):
      x = self.conv2a(input_tensor)
      x = self.bn2a(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2b(x)
      x = self.bn2b(x, training=training)
      x = tf.nn.relu(x)

      x = self.conv2c(x)
      x = self.bn2c(x, training=training)

      x += self.bn2d(self.conv2d(input_tensor))

      return tf.nn.relu(x)
