import tensorflow as tf

# placeholders and feed_dict
# 2 phases of tensorflow: assemble the graph, use session to execute operation in the graph

# used to assemble the graphs first without knowing the values needed for computation

# tf.placeholder(dtype, shape=None, name=None) # dtype is required for data type of the value of placeholder
# https://www.tensorflow.org/api_guides/python/io_ops#Placeholders

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b # short for tf.add(a, b)

# If we try to fetch c, we will run into error
# with tf.Session() as sess:
#   print(sess.run(c))

# NameError
# beacuse to compute c we need value of a while a is just a placeholder. Feed it some value

with tf.Session() as sess:
  # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
  # fetch value of c
  writer = tf.summary.FileWriter('../graphs', sess.graph)
  print(sess.run(c, {a: [1, 2, 3]}))

writer.close()
# [6. 7. 8.]

# to feed multiple data points, just iterate over the data set and feed in the value one at a time
with tf.Session() as sess:
  for a_value in list_of_a_values:
    print(sess.run(c, {a: a_value}))

# you can feed values to tensors that aren't placeholder. Just check if they are feedable. tf.Graph.is_feedable(tensor)

# create ops, tensors etc using the default graph
a = tf.add(2, 5)
b = tf.multiply(a, 3)

# start up a `Session` using the default graph
sess = tf.Session()

# define a dictionary that says to replace the value of `a` with 15
replace_dict = {a: 15}

# Run the session, passing in `replace_dict` as the value to `feed_dict`
sess.run(b, feed_dict=replace_dict) # 45
