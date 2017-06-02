from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# top
X1 = [[0] for i in range(2)]
Y1 = [[0] for i in range(2)]

# bottom
X2 = [0]
Y2 = [0]
n_rewards = 2
plt.ion()
fig, axs = plt.subplots(2,sharex=True)
ax1,ax2 = axs
liner = [*ax1.plot(X1[0], Y1[0], 'r-',lw=.5),
         *ax1.plot(X1[1], Y1[1],'b-',lw=.5)]
linel, = ax2.plot(X2, Y2,'g-', lw=.5)
window_len = 101
linesg1 = ax1.plot(X1[0], Y1[0], 'r-', X1[1], Y1[1], 'b-')
linesg2 = ax2.plot(X2[0], Y2[0], 'g-')
plt.show()

def plot_horz(y):
  ax1.axhline(y=y)

def plot1(x,y,i):
  # append datapoint (x,y) to reward plot i
  assert i in range(n_rewards)
  X1[i].append(x)
  Y1[i].append(y)
  liner[i].set_data(X1[i], Y1[i])
  if len(Y1[i])>window_len:
    linesg1[i].set_data(X1[i], savgol_filter(Y1[i],window_len,2))
  ax1.set_xlim(min([min(X1[j]) for j in range(len(X1))])-1,
               max([max(X1[j]) for j in range(len(X1))])+1)
  ax1.set_ylim(min([min(Y1[j][-100:]) for j in range(len(Y1))])-1,
               max([max(Y1[j][-100:]) for j in range(len(Y1))])+1)
  fig.canvas.draw()

def save_reward_plot(fname):
  fig.savefig("figs/"+fname)

def plot2(x,y):
  X2.append(x)
  Y2.append(y)
  linel.set_data(X2, Y2)
  if len(Y2)>window_len:
    linesg2[0].set_data(X2, savgol_filter(Y2,window_len,2))
  ax2.set_xlim(min(X2)-30,max(X2)+30)
  ax2.set_ylim(min(Y2[-100:])-30,max(Y2[-100:])+30)
  fig.canvas.draw()

