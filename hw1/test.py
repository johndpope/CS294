import numpy as np
import tf_util as tfu
import tensorflow as tf
import model as Model

N = 10000
m = 4
dim_O = (m)
dim_U = (1)
O = tfu.get_placeholder("O",tf.float32, [None, m])
U = tfu.get_placeholder("U",tf.float32, [None, 1])

loss = Model.l2loss_op(.01)
min_opt_fn = tf.train.AdamOptimizer(learning_rate=0.01).minimize
model_fn = Model.model1
batch_size = 16

U_hat = model_fn(O, sum(dim_U))
loss_op = loss(U_hat,U)
opt_op = min_opt_fn(loss_op)

nondata_inputs = []
data_inputs = [O,U]
step_outputs = [opt_op,loss_op]
step_fn = tfu.mem_friendly_function(nondata_inputs, data_inputs,
                                    step_outputs, batch_size)
eval_inputs = [O]
eval_outputs = [U_hat]
action_fn = tfu.mem_friendly_function(nondata_inputs, eval_inputs,
                                      eval_outputs, 1)

module = tfu.module("module1")
step_fn = module(step_fn)
action_fn = module(action_fn)

Xtr = np.random.randn(N,m)
ytr = np.sum(Xtr, 1) + np.random.randn(N)

with tf.Session():
  tfu.initialize()
  
