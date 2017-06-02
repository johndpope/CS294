import tensorflow as tf
import tf_util as tfu

def model1(O, dim_u):
  """
  Assume X/Y are 2d observation/action tensors respectively
  """
  nH1 = 128  # hidden variables in 1st layer
  nH2 = 64
  H1 = tfu.dense(O,nH1,"W0",weight_init=tf.random_normal_initializer(0, 0.1),
                 bias=True)
  H1 = tf.tanh(H1)# tfu.lrelu(H1,leak=.01)
  H2 = tfu.dense(H1,nH2,"W1",weight_init=tf.random_normal_initializer(0, 0.1),
                 bias=True)
  H2 = tf.tanh(H2)
  U_hat = tfu.dense(H2,dim_u,"U_hat",
                    weight_init=tf.random_normal_initializer(0, 0.1),
                    bias=True)
  return U_hat

def l2loss_op(out_hat, out):
  return tf.nn.l2_loss(out_hat-out)

def get_reg_loss(l1_weight):
  def l1regloss_op(out_hat, out):
    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    r1 = l1_weight*sum(tf.reduce_sum(tf.abs(t)) for t in t_vars)
    return tf.nn.l2_loss(out_hat-out) + r1
  return l1regloss_op

def optim_op(model, loss_fn, min_opt_fn):
  def get_opt(inputs, outputs):
    outputs_ = model(*inputs)
    loss = loss_fn(outputs, outputs_)
    return min_opt_fn(loss)
  return get_opt
