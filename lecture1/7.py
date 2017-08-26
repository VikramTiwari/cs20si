# don't do this. building graphs by hand is useless and not practical since no benefit
import tensorflow as tf

g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
  a = tf.constant(3)
# add ops to the user created graph
with g2.as_default():
  b = tf.constant(5)
