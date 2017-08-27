import tensorflow as tf
# Math Operations https://www.tensorflow.org/api_guides/python/math_ops#Arithmetic_Operators
# similar to NumPy

a = tf.constant([3, 6])
b = tf.constant([2, 2])

tf.add(a, b) # [5 8]
tf.add_n([a, b, b]) # [7 10]. Equivalent to a + b + b
tf.multiply(a, b) # [6 12] because multiply is element wise operation
tf.matmul(a, b) # ValueError
tf.matmul(tf.reshape(a, shape=[1, 2]), tf.reshape(b, shape=[2, 1])) # [[18]]
tf.div(a, b) # [1 3]
tf.mod(a, b) # [1 0]
