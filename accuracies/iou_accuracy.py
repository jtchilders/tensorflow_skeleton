import tensorflow as tf


def get_accuracy(pred,label,config):

   # pred shape [batch,points,classes]
   # label shape [batch,points]

   pred = tf.nn.softmax(pred)
   label_onehot = tf.one_hot(label,len(config['data']['classes']),axis=-1)
   return IoU_coeff(pred,label_onehot)


def IoU_coeff(pred,label,smooth=1,point_axis=1):
   intersection = tf.math.reduce_sum(tf.abs(label * pred),axis=point_axis)  # BxC
   # tf.print(intersection,output_stream=sys.stderr)
   union = tf.math.reduce_sum(label,axis=point_axis) + tf.math.reduce_sum(pred,axis=point_axis) - intersection
   # tf.print(union,output_stream=sys.stderr)
   iou = tf.math.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
   for i in range(iou.get_shape()[0].value):
      tf.compat.v1.summary.scalar('accuracy/class_%i' % i,iou[i])
   return tf.math.reduce_mean(iou)
