import tensorflow as tf

# Optimizers: https://www.tensorflow.org/api_guides/python/train#Optimizers
# Autodiff: tf does auto differentiation and updates the value of w and b to minimize loss based on the optimizer

global_step = tf.Variable(0, trainable=False, dtype=tf.int32) # in order to not train a variable set trainable to False
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)

increment_step = global_step.assign_add(1)
optimizer = tf.GraidentDescentOptimizer(learning_rate) # learning rate can be a tensor

# tf.Variable(initial_value=None, trainable=True, collection=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)

# batches
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

X_batch, Y_batch = mnist.test.next_batch(batch_size)
sess.run(train_op, feed_dict={X: X_batch, Y: Y_batch})
