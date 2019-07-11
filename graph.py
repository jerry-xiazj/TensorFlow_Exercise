import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g1 = tf.Graph()
with g1.as_default():
    v = tf.compat.v1.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.compat.v1.get_variable("v", shape=[1], initializer=tf.ones_initializer)

with tf.compat.v1.Session(graph=g1) as sess:
    tf.compat.v1.global_variables_initializer().run()
    with tf.compat.v1.variable_scope("", reuse=True):
        print(sess.run(tf.compat.v1.get_variable("v")))

with tf.compat.v1.Session(graph=g2) as sess:
    tf.compat.v1.global_variables_initializer().run()
    with tf.compat.v1.variable_scope("", reuse=True):
        print(sess.run(tf.compat.v1.get_variable("v")))
