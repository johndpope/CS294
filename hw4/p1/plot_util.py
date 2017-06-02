from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# rewards
Xr = [[0] for i in range(2)]
Yr = [[0] for i in range(2)]

# loss
Xl = [0]
Yl = [0]
n_rewards = 2
plt.ion()
fig, axs = plt.subplots(2,sharex=True)
ax1,ax2 = axs
liner = [*ax1.plot(Xr[0], Yr[0], 'r-',lw=.5),
         *ax1.plot(Xr[1], Yr[1],'b-',lw=.5)]
linesg = ax1.plot(Xr[0], Yr[0], 'r-', Xr[1], Yr[1], 'b-')
linel, = ax2.plot(Xl, Yl,'g-')
plt.show()

def plot_horz(y):
  ax1.axhline(y=y)

def plot_reward(x,y,i):
  # append datapoint (x,y) to reward plot i
  assert i in range(n_rewards)
  Xr[i].append(x)
  Yr[i].append(y)
  liner[i].set_data(Xr[i], Yr[i])
  window_len = 31
  if len(Yr[i])>window_len:
    linesg[i].set_data(Xr[i], savgol_filter(Yr[i],window_len,2))
  ax1.set_xlim(min([min(Xr[j]) for j in range(len(Xr))])-30,
               max([max(Xr[j]) for j in range(len(Xr))])+30)
  ax1.set_ylim(max(min([min(Yr[j]) for j in range(len(Yr))]), -1500)-30,
               max([max(Yr[j]) for j in range(len(Yr))])+30)
  fig.canvas.draw()

def save_reward_plot(fname):
  fig.savefig("figs/"+fname)

def plot_loss(x,y):
  Xl.append(x)
  Yl.append(y)
  linel.set_data(Xl, Yl)
  ax2.set_xlim(min(Xl)-30,max(Xl)+30)
  ax2.set_ylim(min(Yl)-30,max(Yl)+30)
  fig.canvas.draw()


