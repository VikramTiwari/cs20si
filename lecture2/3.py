import tensorflow as tf

# constant of 1d tensor (vector)
a = tf.constant([2, 2], name="vector")

# constant of 2x2 tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="b")

###############
tf.zeros(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are zeros
tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]
###############
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# create a tensor of shape and type (unless type is specified) as the input_tensor but all the elements are zeros

# input tensor is [[0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]
###############
tf.ones(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are ones
tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]
###############
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
# create a tensor of shape and type (unless type is specified) as the input_tensor but all the elements are ones

# input tensor is [[0, 1], [2, 3], [4, 5]]
tf.ones_like(input_tensor) ==> [[1, 1], [1, 1], [1, 1]]
###############
tf.fill(dims, value, name=None)
# create a tensor filled with a scalar value

tf.fills([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
###############
tf.linspace(start, stop, num, name=None)
# create a sequence of num evenly-spaced values are generated beginning at start. If num > 1, the values in sequence increase by stop - start / num - 1, so that the last one is exactly stop.
# start, stop, num must be scalars
# comparable to but slightly different from numpy.linspace
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

tf.linspace(10.0, 13.0, 4, namespace="linspace") ==> [10.0 11.0 12.0 13.0]
###############
tf.range(start, limit=None, delta=1, dtype=None, name='range')
# create a sequence of numbers that begins at start and extends by increments of delta up to but not including limit
# slight differnet from range in Python

# start is 3, limit is 18, delta is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

# start is 3, limit is 1, delta is -0.5
tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]

# limit is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]

###############
# tensorflow sequences are not iterable (NumPy's are)
for _ in np.linspace(0, 10, 4): # OK
for _ in tf.linspace(0, 10, 4): # TypeError("'Tensor' object is not iterable.")

for _ in range(4): # OK
for _ in tf.range(4): # TypeError("'Tensor' object is not iterable.")

################
# genrate random constants from certain distributions
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.mulinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
