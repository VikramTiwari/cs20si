import tensorflow as tf
# Variables

# Constant is constant. Variable can be assigned and changed values
# A constant's value is stored in the graph and it's value is replicated wherever the graph is loaded. A variable is stored separately, and may live on a parameter server. (useful)

my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def())

# output
# node {
#   name: "my_const"
#   op: "Const"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_FLOAT
#     }
#   }
#   attr {
#     key: "value"
#     value {
#       tensor {
#         dtype: DT_FLOAT
#         tensor_shape {
#           dim {
#             size: 2
#           }
#         }
#         tensor_content: "\000\000\200?\000\000\000@"
#       }
#     }
#   }
# }
# versions {
#   producer: 24
# }

# declare variables
# create variable a with scalar value
a = tf.Variable(2, name="scalar")
# create variable b as a vector
b = tf.Variable([2, 3], name="vector")
# create variable c as 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# create variable W as 784x10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784, 10]))

# variable ops
x = tf.Variable(...)

x.initializer # init
x.value() # read op
x.assign(...) # write op
x.assign_add(...)
# and more
# Initilize the variable before using them or run into error (FailedPreconditionError: Attempting to use uninitialized value tensor)

# initialize all variables at once
init = tf.global_variables_initializer()

with tf.Session() as sess:
  tf.run(init) # to run the initializer, not fetching any value

# initialize only as subset of variables
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
  tf.run(init_ab)

# intialize each variable separately
# create variable W as 784x10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
  tf.run(W.initializer)


# Evaluate values of variables
# W is a random 700x100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W)

# Tensor("Variable/read:0", shape=(700, 10), dtype=float32)

# to get value of a variable, evaluate it using eval()
# W is a random 700x100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W.eval())

# Assign values to variables
W =  tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W.eval()) # 10

# why 10 and not 100? W.assign(100) doesn't assign the value 100, but creates an assign op. Run the op in session
W = tf.Variable(10)
assign_op = W.assign(100) # assign also initializes since initializer is the assign operation but with intial value of the variable
with tf.Session() as sess:
  sess.run(assign_op)
  print(W.eval()) # 100
