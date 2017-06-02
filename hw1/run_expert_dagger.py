#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --max_timesteps 1000 --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util as tfu
import gym
import load_policy
import model as Model
import plot_util as putl
import data_util as dutl

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    # parser.add_argument('--expert_interval', type=int, default=2,
    #                     help='have expert run each expert_interval')
    parser.add_argument('--n_opt_steps', type=int, default=2,
                        help='number of iterations over data')
    parser.add_argument('--print_progress', action='store_true')
    args = parser.parse_args()
    
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    dim_U = env.action_space.shape
    dim_O = env.observation_space.shape

    U = tfu.get_placeholder("U",tf.float32, [None, *dim_U])
    O = tfu.get_placeholder("O",tf.float32, [None, *dim_O])

    loss = Model.get_reg_loss(.001)
    min_opt_fn = tf.train.AdamOptimizer(learning_rate=0.0001).minimize
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
    returns = []
    # imitator_returns = []
    observations = []
    used_obs = []
    expert_actions = []
    used_acns = []

    obs_2nrm, obs_dnrm = None, None
    acn_2nrm, acn_dnrm = None, None

    with tf.Session():
        tfu.initialize()
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            testl2 = []         # all test losses
            steps = 0
            while not done:
                expert_action = policy_fn(obs[None,:])
                if i>0:
                    imitation_action = acn_dnrm(
                        action_fn(obs_2nrm(obs[None,:])[None,:]))
                    testl2.append(np.square(expert_action-imitation_action))
                if i == 0:
                    action = expert_action
                else:
                    action = imitation_action
                observations.append(obs)
                expert_actions.append(expert_action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if args.print_progress:
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            if i == 0:
                if args.print_progress:
                    print("total reward expert iteration",i," = ", totalr)
                putl.plot_reward(i,totalr,0)
            else:
                if args.print_progress:
                    print("total reward imitator iteration",i," = ", totalr)
                putl.plot_reward(i,totalr,1)
            returns.append(totalr)
            
            len_train = 1e4
            if len(observations)<len_train:
                observations.extend(used_obs)
                used_obs = []
                expert_actions.extend(used_acns)
                used_acns = []
            # this needs to be done properly:
            train_obs,observations = np.split(observations,[1e4])
            observations = observations.tolist()
            train_acns,expert_actions = np.split(expert_actions,[1e4])
            expert_actions = expert_actions.tolist()
            obs_2nrm, obs_dnrm = dutl.normalize(train_obs)
            acn_2nrm, acn_dnrm = dutl.normalize(train_acns)
            for j in range(args.n_opt_steps):
                _, loss_val_train = step_fn(
                    obs_2nrm(train_obs),
                    acn_2nrm(train_acns))
            used_obs.extend(train_obs.tolist())
            used_acns.extend(train_acns.tolist())
            # loss_total = None
            # if not loss_total:
            #     loss_total = loss_val
            # else:
            #     loss_total += loss_val
            # print("average loss on actions", loss_total/(3))
            if i>0:
                loss_acn = np.sqrt(np.mean(testl2))
                # plot log loss
                putl.plot_loss(i,-np.log(loss_acn))
                print("average loss on actions", loss_acn)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # print(loss_total)
        # expert_data = {'observations': np.array(observations),
        #                'actions': np.array(expert_actions)}


if __name__ == '__main__':
    main()
