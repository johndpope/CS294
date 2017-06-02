from frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()
print(env.__doc__)

# Some basic imports and setup
import numpy as np, numpy.random as nr, gym
np.set_printoptions(precision=3)
def begin_grading():
  print("\x1b[43m")

def end_grading(): 
  print("\x1b[0m")

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()
for t in range(100):
  env.render()
  a = env.action_space.sample()
  ob, rew, done, _ = env.step(a)
  if done:
    break

assert done
env.render();

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)

mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)

print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, which transitions to itself with probability 1 and reward 0.")
print("P[5][0] =", mdp.P[5][0], '\n')

from itertools import product
def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == n
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function, which is zero
    R = np.zeros(shape=(mdp.nS,mdp.nA), dtype = np.float32)
    pis = []
    Rsa_ix = list(product(range(mdp.nS), range(mdp.nA)))
    pi = None
    for it in range(nIt):
        Vprev = Vs[-1]
        # YOUR CODE HERE
        R = np.reshape(np.array(list(
            map(lambda sa: sum(map(lambda x: x[0]*(x[2]+gamma*Vprev[x[1]]),
                                   mdp.P[sa[0]][sa[1]])), Rsa_ix))), 
                       (mdp.nS, mdp.nA))
        # Your code should define variables V: the bellman backup applied to Vprev
        # and pi: the greedy policy applied to Vprev
        oldpi = pi
        pi = np.argmax(R,1)
        V = np.array(list(map(lambda i: R[i,pi[i]], range(mdp.nS))))
        max_diff = np.abs(V - Vprev).max()
        nChgActions=0 if oldpi is None else (pi != oldpi).sum()+nChgActions
        print("%4i      | %6.5f      | %4s          | %5.3f      | %5.3f   "%(it, max_diff, nChgActions, V[0], V[1]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA=0.95 # we'll be using this same value in subsequent problems
begin_grading()
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
end_grading()

import matplotlib.pyplot as plt
%matplotlib inline
for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')
plt.figure()
plt.plot(Vs_VI)
plt.title("Values of different states");

chg_iter = 58
# YOUR CODE HERE
# Your code will need to define an MDP (mymdp)
# like the frozen lake MDP defined above
GAMMA = .95
nS = 2
nA = 2
P = {0:
     {
       0:[(1, 0, 1)], 
       1:[(.9, 0, 1), (.1, 1, .82)]
     },
     1:{
       0:[(1, 0, 1)],
       1:[(1, 1, 1.01)]
     }}
mymdp = MDP(P,nS,nA)
begin_grading()
Vs, pis = value_iteration(mymdp, gamma=GAMMA, nIt=chg_iter+1)
end_grading()

def compute_vpi(pi, mdp, gamma):
  """
  Inputs:
      mdp: MDP
      gamma: discount factor
  Outputs:
      (value_functions, policies)

  len(value_functions) == nIt+1 and len(policies) == n
  """
  nIt = 1000 # maximum number of value updates
  print("Iteration | max|V-Vprev| | # chg actions | V[0]")
  print("----------+--------------+---------------+---------")
  V = np.zeros(mdp.nS) # list of value functions contains the initial value function, which is zero
  for it in range(nIt):
    Vprev = V
    # YOUR CODE HERE
    V = np.reshape(np.array(list(
        map(lambda s: sum(map(lambda x: x[0]*(x[2]+gamma*Vprev[x[1]]),
                               mdp.P[s][pi[s]])), range(mdp.nS)))), 
                   (mdp.nS, mdp.nA))
    max_diff = np.abs(V - Vprev).max()
    if max_diff < eps_diff:
        return V    
    print("%4i      | %6.5f      | %5.3f      | %5.3f   "%(it, max_diff, V[0], V[1]))
  raise Exception('')

begin_grading()
print(compute_vpi(np.ones(16), mdp, gamma=GAMMA))
end_grading()

Vpi=compute_vpi(pis_VI[15], mdp, gamma=GAMMA)
V_vi = Vs_VI[15]
print("From compute_vpi", Vpi)
print("From value iteration", V_vi)
print("Difference", Vpi - V_vi)

def compute_qpi(vpi, pi, mdp,  gamma):
    # YOUR CODE HERE
    return Qpi

begin_grading()
Qpi = compute_qpi(Vpi, pis_VI[-1], mdp, gamma=0.95)
end_grading()
print("Qpi:\n", Qpi)

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, pi_prev, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis
Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);

for s in range(5):
    plt.figure()
    plt.plot(np.array(Vs_VI)[:,s])
    plt.plot(np.array(Vs_PI)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')
