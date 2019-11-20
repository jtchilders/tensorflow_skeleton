import tensorflow as tf


def get_accuracy(pred,label,config):

   pred = tf.argmax(pred,output_type=tf.int32)
   label = tf.argmax(label,output_type=tf.int32)

   correct = tf.reduce_sum(tf.cast(tf.equal(pred,label),dtype=tf.int32))
   total = tf.size(label)

   accuracy = correct / total

   return accuracy
