import model_nur_Dense
import tensorflow as  tf
import numpy


def lossFunction(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true,dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred,dtype=tf.float32)
    minus1 = tf.constant(-1.0, dtype=tf.float32, shape=y_pred.shape)
    first = tf.math.multiply(y_true, tf.math.log(y_pred+1e-10))
    second = tf.math.multiply(tf.constant(1.0, dtype=tf.float32, shape=y_pred.shape) - y_true,
                              tf.math.log(tf.constant(1.0, dtype=tf.float32, shape=y_pred.shape) - y_pred+1e-10))
    loss = tf.math.multiply(minus1, first + second)
    loss = tf.reduce_mean(loss)
    return loss

y_true= [[0,0],[1,1]]
#y_pred=tf.constant(numpy.random.random(4).reshape((2,2)),dtype=tf.float32)
y_pred= [[0,0],[0,1]]
loss = lossFunction(y_true,y_pred)
print(loss)