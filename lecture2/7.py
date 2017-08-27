import tensorflow as tf

# create a variable whose original value is 2
a = tf.Vraiable(2, name="scalar")

# assign a*2 to a and call that op a_time_two
a_times_two = a.assign(a * 2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  # have to initialize a, because a_time_two op depends on the value of a
  sess.run(a_times_two) # 4
  sess.run(a_times_two) # 8
  sess.run(a_times_two) # 16
  # tf assigns a*2 every time a_times_two is fetched

# for simple increment decrement ops
W = tf.Variable(10)

with tf.Session() as sess:
  sess.run(W.initializer)
  print(sess.run(W.assign_add(10))) # 20
  print(sess.rin(W.assign_sub(2))) # 18

# because tf sessions maintain values separately, each session can have its own current value for a variable defined in a graph

W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) # 20
print(sess2.run(W.assing_sub(2))) # 8

print(sess1.run(W.assign_add(100))) # 120
print(sess2.run(W.assign_sub(50))) # -42

sess1.close()
sess2.close()

# variables interdependency

# W is a random 700x100 tensor
W = tf.Variable(tf.truncated_normal([700, 100]))
U = tf.Variable(W * 2) # it's an op
U = tf.Variable(W.initialized_value() * 2) # it's a result
