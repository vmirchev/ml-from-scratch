import random
import numpy as np

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)

def one_hot_encode(x, num_classes: int):
  x = np.asarray(x, dtype=int)
  if x.ndim != 1:
      raise ValueError("Input labels must be a 1D array.")
  if np.any(x < 0) or np.any(x >= num_classes):
      raise ValueError("Labels out of range for one-hot encoding.")
  return np.eye(num_classes, dtype=np.float32)[x]
  
def create_batches(X, y, batch_size=32, shuffle=True):
  if batch_size <= 0:
    raise ValueError("batch_size must be > 0")
  if len(X) != len(y):
    raise ValueError("X and y must have the same number of samples")

  num_samples = X.shape[0]
  if shuffle:
    # shuffle the indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

  for i in range(0, num_samples, batch_size):
    yield X[i:i + batch_size], y[i:i + batch_size]