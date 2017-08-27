import tensorflow as tf

# lazy loading (trap?)
# defer create an op until you need to compute it

# normal loading: create the op z when you assemble the graph

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
  writer = tf.summary.FileWriter('../graphs', sess.graph)
  sess.run(tf.global_variables_initializer())
  for _ in range(10):
      sess.run(z)

writer.close()
# 1 entry for add op is in the graph

# lazy loading

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
  writer = tf.summary.FileWriter('../graphs', sess.graph)
  sess.run(tf.global_variables_initializer())
  for _ in range(10):
    sess.run(tf.add(x, y)) # create the op and only when you need to compute it

writer.close()

# 10 entries for add op are in the graph
# graph definition becomes bloated, slow to load, and expensive to pass around

# to avoid
# always separate teh definition of ops and their execution
# If it can't be avoided follow this appraoch: http://danijar.com/structuring-your-tensorflow-models/
