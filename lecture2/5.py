import tensorflow as tf
# Data Types

# Python native types
# converts native types to tensor types
t_0 = 19  # Treated as 0-d tensor, or "scalar"
tf.zeros_like(t_0)  # 0
tf.ones_like(t_0)  # 1

t_1 = [b"apple", b"peach", b"grape"]  # treated as a 1-d tensor, or "vector"
tf.zeros_like(t_1)  # ['', '', '']
tf.ones_like(t_1)  # TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]  # treated as a 2-d tensor, or "matrix"

tf.zeros_like(t_2)  # 2x2 tensor, all elements are False
tf.ones_like(t_2)  # 2x2 tensor, all elements are True

# Tensorflow native types
# https://www.tensorflow.org/programmers_guide/dims_types

# NumPy data types
# Most of the times TensorFlow types and NumPy types can be used interchangeably

# Python data types might be easy but not great when it comes to handling the data because they lack the ability to explicity state the data type, thus tensorflow has to infer, which could go wrong/incompatible in various cases (complex numbers)
# NumPy data types can be used interchangably but since NumPy is used to create ndarrays, it can not create tensor functions and auto compute derivatives, nor GPU support. In time, it can evolve to the point of incompatibility
