import numpy as np
from .layers import BaseLayer

class ReLU(BaseLayer):
  
  def forward(self, x):
    self.x = x
    return np.maximum(0, x)

  def backward(self, dout):
    dx = dout * (self.x > 0.0)
    return dx
    
def softmax(x):
  x = np.asarray(x, dtype=np.float64)
  if x.ndim == 1:
    x = x.reshape(1, -1)

  shifted_logits = x - np.max(x, axis=1, keepdims=True)
  exp_logits = np.exp(shifted_logits)
  return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
