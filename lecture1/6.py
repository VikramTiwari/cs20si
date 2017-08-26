import tensorflow as tf

x = 2
y = 3

add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
  z, not_useless = sess.run([pow_op, useless])
  # tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
  # pass all variables whose values you want to a list in fetches
