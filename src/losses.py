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