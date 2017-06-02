import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from tensorflow.python.ops import random_ops
import pdb
import plot_util as pltu


def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant_initializer(out)
    return _initializer

def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
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

def NonLinearValueFunction(X, dim_out):
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 64, "h1_nlvf", 
                             weight_init=tf.random_normal_initializer(1)))\
        # hidden layer
    sy_h2 = tf.nn.relu(dense(sy_h1, 32, "h2_nlvf",
                             weight_init=tf.random_normal_initializer(.5)))
    return dense(sy_h2)

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
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the\
        # advantage function to avoid introducing bias
        logz.dump_tabular()

# models for mean and stddev computation of action
def model1(sy_ob_no, ac_dim):
    sy_mean_h1 = tf.nn.relu(dense(sy_ob_no, 128, "mean_h1",
                                  weight_init=tf.random_normal_initializer(1)))\
        # hidden layer
    sy_mean_na = dense(sy_mean_h1, ac_dim, "mean", weight_init=tf.random_normal_initializer(0.05))\
        # mean control output
    sy_std_na = tf.nn.softplus(dense(sy_ob_no, ac_dim, "std",
                                     weight_init=tf.random_normal_initializer(.05)))\
        # parametrized stddev of control output
    return sy_mean_na, sy_std_na

# models for mean and stddev computation of action
def model2(sy_ob_no, ac_dim):
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 128, "h1", weight_init=tf.random_normal_initializer(1)))\
        # hidden layer
    sy_mean_na = dense(sy_h1, ac_dim, "mean", weight_init=tf.random_normal_initializer(0.05))\
        # mean control output
    sy_std_na = tf.nn.softplus(dense(sy_h1, ac_dim, "std",
                                     weight_init=tf.random_normal_initializer(mean=.5, stddev=.5)))\
        # parametrized stddev of control output
    return sy_mean_na, sy_std_na

def model3(sy_ob_no, ac_dim):
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 128, "h1", weight_init=tf.random_normal_initializer(1)))\
        # hidden layer
    sy_h2 = tf.nn.relu(dense(sy_h1, 64, "h2",
                             weight_init=tf.random_normal_initializer(.5)))
    sy_mean_na = dense(sy_h2, ac_dim, "mean", weight_init=tf.random_normal_initializer(0.05))\
        # mean control output
    sy_std_na = tf.nn.softplus(dense(sy_h2, ac_dim, "std",
                                     weight_init=tf.random_normal_initializer(mean=.5, stddev=.5)))\
        # parametrized stddev of control output
    return sy_mean_na, sy_std_na



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

    sy_logprob_n = pdf_ac / (sy_std_na*(cdf_beta-cdf_alpha))
        # log-prob of actions taken -- used for policy gradient calculation

    return sy_logprob_n


def main_pendulum(n_iter=1000, gamma=.99, min_timesteps_per_batch=1000,
                  stepsize=1e-4, animate=False, logdir=None):
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)\
        # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.float32) \
        # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) \
        # advantage function estimate
    sy_mean_na, sy_std_na = model3(sy_ob_no, ac_dim)

    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)\
        # mean BEFORE update (just used for KL diagnostic)
    sy_oldstd_na = tf.placeholder(shape=[None, ac_dim], name='oldstd', dtype=tf.float32)\
        # std BEFORE update (just used for KL diagnostic)

    sy_n = tf.shape(sy_ob_no)[0]  # batch sys.getsizeof()

    if ac_dim == 1:
        sy_mean_na = tf.squeeze(sy_mean_na)
        sy_std_na = tf.squeeze(sy_std_na)
        sy_oldmean_na = tf.squeeze(sy_oldmean_na)
        sy_oldstd_na = tf.squeeze(sy_oldstd_na)
        sy_ac_n = tf.squeeze(sy_ac_n)

        sy_sampled_ac = random_ops.parameterized_truncated_normal(shape=[sy_n],
                                                                  means=sy_mean_na,
                                                                  stddevs=sy_std_na,
                                                                  minvals=env.action_space.low[0],
                                                                  maxvals=env.action_space.high[0])\
            # sampled actions, used for defining the policy (NOT computing the policy gradient)

        ds = tf.contrib.distributions
        dist_normal = ds.Normal(loc = sy_mean_na, scale=sy_std_na)
        old_dist_normal = ds.Normal(loc = sy_oldmean_na, scale=sy_oldstd_na)

        sy_logprob_n = truncated_normal_logprob(dist_normal, sy_ac_n,sy_std_na,
                                                lb=env.action_space.low[0],
                                                ub=env.action_space.high[0])
            # log-prob of actions taken -- used for policy gradient calculation
    else:
        ValueError("currently only works w/ 1-d actions")


    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC
    # PURPOSES >>>>
    # IK: kl and ent for normal dist, not truncated normal (that we sample from)
    sy_kl = tf.reduce_sum(ds.kl(old_dist_normal,dist_normal)) / tf.to_float(sy_n)
    sy_ent = tf.reduce_sum(dist_normal.entropy()) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n)\
        # Loss function that we'll differentiate to get the policy gradient ("surr" is for\
        # "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)\
        # Symbolic, in case you want to change the stepsize during optimization. (We're not doing\
        # that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
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
                ob = np.squeeze(ob)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
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
        _, oldmean_na, oldstd_na = sess.run([update_op, sy_mean_na, sy_std_na],
                                            feed_dict={sy_ob_no:ob_no, sy_ac_n:np.squeeze(ac_n),
                                                       sy_adv_n:standardized_adv_n,
                                                       sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no,sy_oldmean_na: oldmean_na,
                                                       sy_oldstd_na:oldstd_na})

        # Log diagnostics
        eprewmean = np.mean([path["reward"].sum() for path in paths])
        evafter = explained_variance_1d(vf.predict(ob_no), vtarg_n)
        pltu.plot_reward(total_timesteps,eprewmean,0)
        pltu.plot_loss(total_timesteps,evafter)
        logz.log_tabular("EpRewMean", eprewmean)
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("acMean", np.mean(ac_n))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", evafter)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the\
        # advantage function to avoid introducing bias
        logz.dump_tabular()


if __name__ == "__main__":
    main_pendulum(logdir=None) # when you want to start collecting results, set the logdir
