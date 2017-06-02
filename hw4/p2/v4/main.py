import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

# from tensorflow example
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant_initializer(out)
    return _initializer

def dense(x, size, name, weight_init=None, bias_init=tf.zeros_initializer()):
    """
    Dense (fully connected) layer
    """
    with tf.name_scope(name+'/w'):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
        variable_summaries(w)
    with tf.name_scope(name+'/b'):
        b = tf.get_variable(name + "/b", [size], initializer=bias_init)
        variable_summaries(b)
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NonLinearValueFunction(object):
    def model(self,X):
        with tf.variable_scope("value_estimator"):
            sy_h1 = tf.nn.relu(dense(X, 64, "h1", weight_init=tf.random_normal_initializer(1)))\
                # first hidden layer
            sy_h2 = tf.nn.relu(dense(sy_h1, 32, "h2", weight_init=tf.random_normal_initializer(.5)))
            val = tf.squeeze(dense(sy_h2, 1, "val", weight_init=tf.random_normal_initializer(.05)))
        return val
    def __init__(self, _nsteps=1):
        self.nsteps = _nsteps   # number of gradient steps to run each iteration
    def set_net_vars(self, _sess, sy_X, sy_y, optimizer=None, stepsize=1e-3, reg_coeff=0):
        self.sess = _sess
        self.X = sy_X
        self.y = sy_y
        self.y_pred = self.model(self.X)
        self.loss = tf.reduce_mean(tf.square(self.y_pred-self.y))
        objective = self.loss
        if not optimizer:
            optimizer = tf.train.AdamOptimizer
        if reg_coeff > 0:
            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_estimator")
            self.reg = sum(tf.reduce_mean(tf.abs(t)) for t in t_vars)  # regularization
            objective += reg_coeff*self.reg
        self.update = optimizer(stepsize).minimize(objective)
        with tf.variable_scope('performance/value'):
            with tf.name_scope('loss'):
                tf.summary.scalar('value',objective)
    def predict(self,X):
        return self.sess.run(self.y_pred, feed_dict={self.X:X})
    def fit(self,X,y):
        for i in range(self.nsteps):
            self.sess.run(self.update, feed_dict={self.X:X, self.y:y})
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

def main_cartpole(n_iter=100, gamma=1.0, min_timesteps_per_batch=1000,
                  stepsize=1e-2, animate=True, logdir=None):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)\
        # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) \
        # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) \
        # advantage function estimate
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 32, "h1", weight_init=tf.random_normal_initializer(1.0)))\
        # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final",
                         weight_init=tf.random_normal_initializer(0.05)) \
        # "logits", describing probability distribution of final layer\
        # we use a small initialization for the last layer, so the initial policy has maximal\
        # entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32)\
        # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions

    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0]\
        # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]  # batch size
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n)\
        # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC
    # PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na)
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum(- sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n)\
        # Loss function that we'll differentiate to get the policy gradient ("surr" is for\
        # "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)\
        # Symbolic, in case you want to change the stepsize during optimization. (We're not doing\
        # that currently)
    update_op = tf.train.AdamOptimizer(stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`

    if type(vf) == NonLinearValueFunction:
        sy_rew_n = tf.placeholder(shape=[None], name="rew", dtype=tf.float32) \
            # batch of rewards given by the policy, used for value function gradient computation
        vf.set_net_vars(sess, sy_ob_no, sy_rew_n)
    tf.initialize_all_variables().run() # pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na],
                                   feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n,
                                              sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no,
                                                       sy_oldlogits_na:oldlogits_na})
        # Log diagnostics
        eprewmean = np.mean([path["reward"].sum() for path in paths])
        evbefore = explained_variance_1d(vpred_n, vtarg_n)
        evafter = explained_variance_1d(vf.predict(ob_no), vtarg_n)
        # pltu.plot_reward(total_timesteps,eprewmean,0)
        # pltu.plot_loss(total_timesteps,evbefore)
        logz.log_tabular("EpRewMean", eprewmean)
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", evbefore)
        logz.log_tabular("EVAfter", evafter)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the\
        # advantage function to avoid introducing bias
        logz.dump_tabular()

# models for mean and stddev computation of action
def model1(sy_ob_no, ac_dim):
    sy_mean_h1 = tf.nn.relu(dense(sy_ob_no, 128, "mean_h1",
                                  weight_init=tf.random_normal_initializer(1)), name="mean_h1_a")\
        # hidden layer
    w_init = tf.random_normal_initializer(0.05)
    b_init = tf.random_normal_initializer(0.05)
    sy_mean_na = dense(sy_mean_h1, ac_dim, "mean", weight_init=w_init, bias_init=b_init)\
        # mean control output
    w_init = tf.random_normal_initializer(0.05)
    b_init=tf.random_normal_initializer(mean=.5, stddev=.1)
    sy_std_na = tf.nn.softplus(dense(sy_ob_no, ac_dim, "std", weight_init=w_init,
                                     bias_init=b_init)) # parametrized stddev of control output
    return tf.squeeze(sy_mean_na), tf.squeeze(sy_std_na)

# models for mean and stddev computation of action
def model2(sy_ob_no, ac_dim):
    w_init=tf.random_normal_initializer(stddev=1)
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 64, "h1", weight_init=tf.random_normal_initializer(1)))\
        # hidden layer
    w_init=tf.random_normal_initializer(5e-1)
    b_init=tf.random_normal_initializer(5e-1)
    sy_mean_na = dense(sy_h1, ac_dim, "mean", weight_init=w_init, bias_init=b_init)\
        # mean control output
    w_init=tf.random_normal_initializer(stddev=5e-1)
    b_init=tf.random_normal_initializer(mean=1,stddev=5e-1)
    sy_std_na = tf.nn.softplus(dense(sy_h1, ac_dim, "std", weight_init=w_init, bias_init=b_init))\
        # parametrized stddev of control output
    return tf.squeeze(sy_mean_na), tf.squeeze(sy_std_na)

def model3(sy_ob_no, ac_dim):
    w_init=tf.random_normal_initializer(stddev=1e-4)
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 128, "h1", weight_init=w_init))\
        # hidden layer
    w_init=tf.random_normal_initializer(stddev=1e-4)
    sy_h2 = tf.nn.relu(dense(sy_h1, 64, "h2", weight_init=w_init), name="h2_a")
    # with tf.name_scope('h2_a'):
    #   variable_summaries(sy_h2)
    w_init=tf.random_normal_initializer(stddev=1e-2)
    b_init=tf.random_normal_initializer(stddev=.5)
    sy_mean_na = dense(sy_h2, ac_dim, "mean", weight_init=w_init, bias_init=b_init)\
        # mean control output
    b_init=tf.random_normal_initializer(mean=.5, stddev=.1)
    sy_std_na = tf.nn.softplus(dense(sy_h2, ac_dim, "std", weight_init=w_init,
                                     bias_init=b_init))\
        # parametrized stddev of control output
    return tf.squeeze(sy_mean_na), tf.squeeze(sy_std_na)

def truncated_normal_logprob(dist_normal, sy_std_na, sy_ac_n, lb=-2, ub=2):
    """
    Symbolic truncated normal log-prob of sy_ac_n given mean = `sy_mean_na`, std=`sy_std_na`,
    lowerbound `lb`,upperbound `ub`.
    """

    # converting from normal to truncated_normal
    # based on https://en.wikipedia.org/wiki/Truncated_normal_distribution
    cdf_alpha = dist_normal.cdf(tf.constant(lb, dtype = tf.float32))
    cdf_beta = dist_normal.cdf(tf.constant(ub, dtype = tf.float32))
    pdf_ac = dist_normal.prob(sy_ac_n)

    sy_logprob_n = tf.log(pdf_ac / (sy_std_na*(cdf_beta-cdf_alpha)))
        # log-prob of actions taken -- used for policy gradient calculation

    return sy_logprob_n

class NonLinearPolicyFunction(object):
    def __init__(self, sess, sy_ob_no, policy_model, sy_y_old, sy_ac_n, dist_logprobfn_sampler,
                 calc_kl, calc_ent, sy_adv_n, ac_dim, sy_stepsize, optimizer=None, reg_coeff=0,
                 n_steps=1):
        self.sess = sess
        self.sy_ob_no = sy_ob_no
        with tf.variable_scope("policy/predict"):
            self.sy_y_pred = policy_model(sy_ob_no, ac_dim)
        self.sy_n = tf.shape(sy_ob_no)[0]  # batch size
        self.dist, log_prob_fn, sampler = dist_logprobfn_sampler(self.sy_y_pred)
        self.sy_sampled_ac = sampler()
        self.sy_y_old = sy_y_old
        self.old_dist, _, _= dist_logprobfn_sampler(self.sy_y_old)
        self.sy_ac_n = sy_ac_n
        self.sy_adv_n = sy_adv_n
        with tf.variable_scope("policy/vars", reuse=True):
            self.sy_logprob_n = log_prob_fn(sy_ac_n)
            self.sy_loss = self.loss_fn(reg_coeff)
            self.sy_kl = calc_kl(self.old_dist,self.dist)
            self.sy_ent = calc_ent(self.dist)
            with tf.name_scope('acn'):
                variable_summaries(self.sy_ac_n)
            self.sy_stepsize = sy_stepsize
            with tf.name_scope('stepsize'):
                tf.summary.scalar('value',self.sy_stepsize)
        with tf.variable_scope('performance/policy'):
            with tf.name_scope('loss'):
                tf.summary.scalar('value',self.sy_loss)
            with tf.name_scope('kl'):
                tf.summary.scalar('value',self.sy_kl)
            with tf.name_scope('ent'):
                tf.summary.scalar('value',self.sy_ent)
        
        self.nsteps = n_steps

        if not optimizer:
            optimizer = tf.train.AdamOptimizer
        self.update_op = optimizer(self.sy_stepsize).minimize(self.sy_loss)

    def loss_fn(self, reg_coeff):
        """
        Loss function that we'll differentiate to get the policy gradient
        """
        objective = -tf.reduce_mean(self.sy_adv_n * self.sy_logprob_n)
        if reg_coeff>0:
            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy")
            reg = sum(tf.reduce_mean(tf.abs(t)) for t in t_vars)  # regularization
            objective+=reg_coeff*reg
        return objective
        
    def sample_action(self, ob_no):
        """
        samples actions for each ob in ob_no
        """
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob_no})

    def update_policy(self, ob_no, ac_n, standardized_adv_n, stepsize):
        """
        Updates policy and return prediction
        """
        return self.sess.run([self.update_op, self.sy_y_pred],
                             feed_dict={self.sy_ob_no:ob_no, self.sy_ac_n:np.squeeze(ac_n),
                                        self.sy_adv_n:standardized_adv_n,
                                        self.sy_stepsize:stepsize})

    def get_kl_ent(self, ob_no, y_old):
        return self.sess.run([self.sy_kl, self.sy_ent], feed_dict={self.sy_ob_no:ob_no,
                                                                   self.sy_y_old: y_old})

    # def create_board(self):
    #     with tf.name_scope('summaries'):
    #         tf.summary.histogram('ac_mean', self.sy_ac_n)

class Summarizer(object):
    def __init__(self, sess, sy_X, summaries_dir='../../summaries/p2/v6/', name='r6'):
      self.merged = tf.summary.merge_all()
      self.train_writer = tf.summary.FileWriter(summaries_dir + name, sess.graph)
      self.sess = sess
      self.sy_X = sy_X
    def create_summary(self, ix, X):
      if type(self.sy_X) == list:
        assert len(self.sy_X) == len(X)
        feed_dict={self.sy_X[i]:X[i] for i in range(len(self.sy_X))}
      else:
        feed_dict={self.sy_X:X}
      summary = self.sess.run(self.merged, feed_dict=feed_dict)
      self.train_writer.add_summary(summary, ix)

def main_pendulum(n_iter=int(1e6), gamma=.97, min_timesteps_per_batch=3000, stepsize=1e-3,
                  animate=False, logdir=None, desired_kl=2e-3):
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    logz.configure_output_dir(logdir)
    vf = NonLinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)\
        # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.float32) \
        # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) \
        # advantage function estimate
    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)\
        # mean BEFORE update (just used for KL diagnostic)
    sy_oldstd_na = tf.placeholder(shape=[None, ac_dim], name='oldstd', dtype=tf.float32)\
        # std BEFORE update (just used for KL diagnostic)
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)\
        # Symbolic, in case you want to change the stepsize during optimization. (We're not doing\
        # that currently)

    with tf.variable_scope("performance"):
        with tf.name_scope('reward'):
            sy_reward = tf.placeholder(shape=[], dtype=tf.float32) # actual reward placeholder
            tf.summary.scalar('value', sy_reward)
        with tf.name_scope('evbefore'):
            sy_evbefore = tf.placeholder(shape=[], dtype=tf.float32) # actual evbefore placeholder
            tf.summary.scalar('value', sy_evbefore)
        with tf.name_scope('evafter'):
            sy_evafter = tf.placeholder(shape=[], dtype=tf.float32) # actual evafter placeholder
            tf.summary.scalar('value', sy_evafter)
    ds = tf.contrib.distributions

    # only tested for ds.Normal
    # can probably use MultivariateNormalLinearOperator if action is higher dim.
    if ac_dim == 1:
        sy_oldmean_na = tf.squeeze(sy_oldmean_na)
        sy_oldstd_na = tf.squeeze(sy_oldstd_na)
        sy_ac_n = tf.squeeze(sy_ac_n)

        def dist_logprobfn_sampler(y):
            sy_mean, sy_std = y
            dist = ds.Normal(loc = sy_mean, scale=sy_std)

            def log_prob_fn(sy_ac_n):
                return dist.log_prob(sy_ac_n)
                    # log-prob of actions taken -- used for policy gradient calculation
            return dist, log_prob_fn, dist.sample

    else:
        ValueError("currently only works w/ 1-d actions")

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC
    # PURPOSES >>>>
    # IK: kl and ent for normal dist, not truncated normal (that we sample from)
    def calc_kl(old_dist,dist):
      with tf.name_scope('/kl'):
        var = tf.reduce_mean(ds.kl(old_dist,dist))
        # variable_summaries (var)
        return var

    def calc_ent(dist):
      with tf.name_scope('/ent'):
        var = tf.reduce_mean(dist.entropy())
        # variable_summaries(var)
        return var
    # <<<<<<<<<<<<<

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    pf = NonLinearPolicyFunction(sess, sy_ob_no, model3, (sy_oldmean_na, sy_oldstd_na), sy_ac_n,
                                 dist_logprobfn_sampler, calc_kl, calc_ent, sy_adv_n, ac_dim,
                                 sy_stepsize)

    if type(vf) == NonLinearValueFunction:
        sy_rew_n = tf.placeholder(shape=[None], name="rew", dtype=tf.float32) \
            # batch of rewards given by the policy, used for value function gradient computation
        vf.set_net_vars(sess, sy_ob_no, sy_rew_n)

    summary = Summarizer(sess, [sy_ob_no, sy_ac_n, sy_adv_n, (sy_oldmean_na, sy_oldstd_na),
                                sy_reward, sy_evbefore, sy_evafter, sy_stepsize, sy_rew_n])

    tf.initialize_all_variables().run() # pylint: disable=E1101

    total_timesteps = 0

    # pltr = pltu.plotter("plot", 2, [2,1], [101,101])
    # ix_refresh = 1
    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                ob = np.squeeze(ob)
                obs.append(ob)
                ac = pf.sample_action(ob[None])
                if type(ac) != np.ndarray:
                    ac = np.array([ac])
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break
            path = {"observation": np.array(obs), "terminated": terminated,
                    "reward": np.array(rewards), "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = np.squeeze(path["reward"])
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, y_pred = pf.update_policy(ob_no, ac_n, standardized_adv_n, stepsize)

        kl, ent = pf.get_kl_ent(ob_no, y_pred)
    
        if kl > desired_kl * 2:
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2:
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')

        # Log diagnostics
        eprewmean = np.mean([path["reward"].sum() for path in paths])
        evbefore = explained_variance_1d(vpred_n, vtarg_n)
        evafter = explained_variance_1d(vf.predict(ob_no), vtarg_n)

        # with tf.name_scope('reward'):
        #     variable_summaries(sy_reward)
        # with tf.name_scope('evberfore'):
        #     variable_summaries(sy_evberfore)
        # with tf.name_scope('evafter'):
        #     variable_summaries(sy_evafter)

        summary.create_summary(total_timesteps, [ob_no, ac_n, standardized_adv_n, y_pred, eprewmean,
                                                 evbefore, evafter, stepsize, vtarg_n])

        # refresh_lims = False
        # if total_timesteps > ix_refresh*1e5:
        #     ix_refresh += 1
        #     refresh_lims = True
        #     pltr.save_plot("plot_t"+str(total_timesteps)+".png")
        # pltr.plot(total_timesteps,eprewmean,1,0, refresh_lims)
        # pltr.plot(total_timesteps,evbefore,0,0, refresh_lims)
        # pltr.plot(total_timesteps,ent,0,1, refresh_lims)
        logz.log_tabular("EpRewMean", eprewmean)
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("acMean", np.mean(ac_n))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", evbefore)
        logz.log_tabular("EVAfter", evafter)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the\
        # advantage function to avoid introducing bias
        logz.dump_tabular()

if __name__ == "__main__":
    main_pendulum(logdir=None) # when you want to start collecting results, set the logdir
