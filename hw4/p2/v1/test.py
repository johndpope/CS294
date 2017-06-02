import tensorflow as tf
from tensorflow.python.ops import random_ops
tf.reset_default_graph()
ds = tf.contrib.distributions
n = 5
means = random_ops.parameterized_truncated_normal(shape = [n,1],minvals = -2)
stddevs = random_ops.parameterized_truncated_normal(shape = [n,1],minvals = 0)
dist_normal = ds.Normal(loc = means, scale=stddevs)
cdf_alpha = dist_normal.cdf(-2*tf.ones(n))
