import tensorflow as tf


def get_datasets(config_file):

    batch_size = config_file['data']['batch_size']
    shuffle_buffer = config_file['data']['shuffle_buffer']

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.cast(x_train,tf.float32)
    x_test = tf.cast(x_test,tf.float32)
    y_train = tf.cast(y_train,tf.float32)
    y_test = tf.cast(y_test,tf.float32)

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if 'hvd' in config_file and config_file['hvd'] is not None:
        hvd = config_file['hvd']
        train_ds = train_ds.shard(hvd.size(), hvd.rank())
        test_ds = test_ds.shard(hvd.size(), hvd.rank())

    train_ds = train_ds.shuffle(shuffle_buffer).batch(batch_size)
    test_ds  = test_ds.batch(batch_size)

    return train_ds,test_ds
