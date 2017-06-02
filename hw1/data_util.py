import numpy as np

def normalize(X):
  """
  assume X is mx of size n_observations x n_features
  """
  Xs = np.squeeze(X)
  means = np.mean(Xs,0)
  stds = np.std(np.square(Xs),0,ddof=1)

  def de_norm(X_norm):
    return np.squeeze(X_norm)*stds + means

  def to_norm(X_new):
    return (np.squeeze(X_new)-means)/np.maximum(stds, 1e-6)

  return to_norm, de_norm
