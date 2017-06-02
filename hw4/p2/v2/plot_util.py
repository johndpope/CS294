from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter as sgf

plt.ion()
class plotter(object):
  """
  convinience class for plotting 2d streaming data
  `n_curves`: list with number of curves in each subplot
  `sg_len`: list with number of length param of sg filter (one per subplot)
  savgol filters are applied w/ poly of deg 2
  """
  def __init__(self, name, n_subplots, n_curves, sg_len, lims_range=100):
    assert all([l % 2 == 1 for l in sg_len])
    self.name = name
    self.n_curves = n_curves
    self.n_subplots = n_subplots
    self.X = []
    self.Y = []
    self.fig, self.axs = plt.subplots(n_subplots,sharex=True)
    cmap_arr = []
    self.X = [[[] for j in range(n_curves[i])] for i in range(n_subplots)]
    self.Y = [[[] for j in range(n_curves[i])] for i in range(n_subplots)]
    cmap_arr = [plt.cm.get_cmap("hsv", n_curves[i]+1) for i in range(n_subplots)]
    self.lines = [[] for i in range(n_subplots)]
    self.sg_lines = [[] for i in range(n_subplots)]
    for i in range(n_subplots):
      for j in range(n_curves[i]):
        self.lines[i].append(*self.axs[i].plot([], [], c=cmap_arr[i](j), linestyle='-',lw=.3))
        self.sg_lines[i].append(*self.axs[i].plot([], [], c=cmap_arr[i](j), linestyle='-'))
    self.sg_len = sg_len
    self.Xlims = [{"min":0, "max":0} for i in range(n_subplots)]
    self.Ylims = [{"min":0, "max":0} for i in range(n_subplots)]
    self.lims_range = lims_range
    plt.show()

  def plot(self, x, y, i, j, refresh_lims = False):
    assert i in range(self.n_subplots) and j in range(self.n_curves[i])
    self.X[i][j].append(x)
    self.Y[i][j].append(y)
    self.lines[i][j].set_data(self.X[i][j], self.Y[i][j])
    if (len(self.Y[i][j])>self.sg_len[i]):
      self.sg_lines[i][j].set_data(self.X[i][j], sgf(self.Y[i][j], self.sg_len[i],2))

    if refresh_lims:
      self.Xlims[i] = {"min": min([min(self.X[i][k][-self.lims_range:]) for k in
                                   range(self.n_curves[i])]),
                       "max": max([max(self.X[i][k][-self.lims_range:]) for k in
                                   range(self.n_curves[i])])}
      self.Ylims[i] = {"min": min([min(self.Y[i][k][-self.lims_range:]) for k in
                                   range(self.n_curves[i])]),
                       "max": max([max(self.Y[i][k][-self.lims_range:]) for k in
                                   range(self.n_curves[i])])}
      self.axs[i].set_xlim(self.Xlims[i]["min"], self.Xlims[i]["max"])
      self.axs[i].set_ylim(self.Ylims[i]["min"], self.Ylims[i]["max"])

    else:
      if x<self.Xlims[i]["min"]:
        refresh_lims=True
        self.Xlims[i]["min"] = x
      if self.Ylims[i]["max"]<x:
        refresh_lims=True
        self.Xlims[i]["max"] = x
      if y<self.Ylims[i]["min"]:
        refresh_lims=True
        self.Ylims[i]["min"] = y
      if self.Ylims[i]["max"]<y:
        refresh_lims=True
        self.Ylims[i]["max"] = y
      if refresh_lims:
        self.axs[i].set_xlim(self.Xlims[i]["min"], self.Xlims[i]["max"])
        self.axs[i].set_ylim(self.Ylims[i]["min"], self.Ylims[i]["max"])

    self.fig.canvas.draw()

  def save_plot(self,fname):
    self.fig.savefig("figs/"+fname)
