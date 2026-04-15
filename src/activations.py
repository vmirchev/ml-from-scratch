import numpy as np
from layers import BaseLayer

class ReLU(BaseLayer):
  
  def forward(self, x):
    self.x = x
    return np.maximum(0, x)

  def backward(self, dout):
    dx = dout * (self.x > 0.0)
    return dx
