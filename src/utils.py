import random
import matplotlib.pyplot as plt
import numpy as np

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)

def k_fold_split(X, y, k=5, shuffle=True):
  if k <= 1:
    raise ValueError("k must be greater than 1")
  if k > len(X):
      raise ValueError("k cannot be greater than the number of samples")
  if len(X) != len(y):
      raise ValueError("X and y must have the same number of samples")

  num_samples = X.shape[0]

  # similar to create batches we can shuffle data
  indices = np.arange(num_samples)

  if shuffle:
      np.random.shuffle(indices)

  fold_sizes = np.full(k, num_samples // k, dtype=int)
  fold_sizes[:num_samples % k] += 1
  current = 0

  # create folds exactly as in the example images
  for fold_size in fold_sizes:
    start, stop = current, current + fold_size
    val_indices = indices[start:stop]
    train_indices = np.concatenate((indices[:start], indices[stop:]))
    yield train_indices, val_indices
    current = stop  

def plot_training_curves(train_losses, train_accuracies, val_losses=None, val_accuracies=None):
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))

  # plot for Loss
  axes[0].plot(train_losses, label='Train Loss')
  if val_losses is not None:
      axes[0].plot(val_losses, label='Val Loss', linestyle='--')

  axes[0].set_title('Loss Over Epochs')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Loss')
  axes[0].legend()
  axes[0].grid(True)

  # plot for Accuracy
  axes[1].plot(train_accuracies, label='Train Accuracy')
  if val_accuracies is not None:
      axes[1].plot(val_accuracies, label='Val Accuracy', linestyle='--')

  axes[1].set_title('Accuracy Over Epochs')
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Accuracy')
  axes[1].legend()
  axes[1].grid(True)

  plt.tight_layout() # adjust layout to prevent overlapping titles
  plt.show()

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