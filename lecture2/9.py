import tensorflow as tf

# Control dependencies
# to specify the order of independent operations

# graph g has 5 ops: a, b, c, d, e
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed
  d = ...
  e = ...
