from .activations import softmax
import numpy as np

class CrossEntropyLoss:
  EPSILON = 1e-9

  def __init__(self):
    self.y_true = None
    self.softmax_output = None
    self.batch_size = 0

  def forward(self, logits, y_true):
    if logits.shape != y_true.shape:
      raise ValueError(f"logits shape {logits.shape} must match y_true shape {y_true.shape}")

    self.y_true = y_true
    self.batch_size = y_true.shape[0]

    # apply stable softmax
    self.softmax_output = softmax(logits)

    # clip probabilities for numerical stability - avoid log(0)
    clipped_softmax_output = np.clip(self.softmax_output, self.EPSILON, 1.0 - self.EPSILON)

    # calculate cross-entropy loss
    loss = -np.mean(np.sum(self.y_true * np.log(clipped_softmax_output), axis=1))
    return loss

  def backward(self):
    # dL/d(logits) = (softmax_output - y_true) / batch_size
    grad_logits = (self.softmax_output - self.y_true) / self.batch_size
    return grad_logits
    
class BinaryCrossEntropyLoss:
  EPSILON = 1e-9

  def __init__(self):
    self.y_true = None
    self.sigmoid_output = None
    self.batch_size = 0

  def forward(self, logits, y_true):
    # Ensure y_true is a 2D array for consistent operations, typically 0 or 1
    self.y_true = np.asarray(y_true).reshape(-1, 1)
    self.batch_size = self.y_true.shape[0]

    # apply sigmoid activation to get probabilities
    # clipped sigmoid for numerical stability
    self.sigmoid_output = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))

    # clip probabilities for numerical stability
    clipped_sigmoid_output = np.clip(self.sigmoid_output, self.EPSILON, 1.0 - self.EPSILON)

    # calculate binary cross-entropy loss
    loss = -np.mean(self.y_true * np.log(clipped_sigmoid_output) + (1 - self.y_true) * np.log(1 - clipped_sigmoid_output))

    return loss

  def backward(self):
    # dL/d(logits) = (sigmoid_output - y_true) / batch_size
    grad_logits = (self.sigmoid_output - self.y_true) / self.batch_size
    return grad_logits