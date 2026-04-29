from abc import ABC, abstractmethod
import numpy as np

class BaseLayer(ABC):
  is_training: bool
  
  def __init__(self):
    self.is_training = True

  def train(self):
    self.is_training = True

  def eval(self):
    self.is_training = False

  @abstractmethod
  def forward(self, x):
    raise NotImplementedError

  @abstractmethod
  def backward(self, dout):
    raise NotImplementedError

  def step(self, lr):
    pass
    
class LinearLayer(BaseLayer):
  def __init__(self, in_features:int, out_features:int, initialization: str = "he", l2_lambda = 0.0):
    super().__init__()

    if initialization == "he":
      scale = np.sqrt(2.0 / in_features)
    elif initialization == "xavier":
      scale = np.sqrt(1.0 / in_features)
    elif initialization == "standard":
      scale = 0.01
    else:
      raise ValueError(f"Unknown init: {initialization}")

    self.w = np.random.randn(in_features, out_features) * scale
    self.b = np.zeros((1, out_features)) # we need one bias per output feature

    self.l2_lambda = l2_lambda

  def forward(self, x):
    self.x = x
    return x @ self.w + self.b

  def backward(self, dout):
    self.dw = self.x.T @ dout
    self.db = np.sum(dout, axis=0, keepdims=True)

    # l2 regularization aka weight decay
    self.dw += 2 * self.l2_lambda * self.w

    dx = dout @ self.w.T
    return dx

  def step(self, lr):
    self.w = self.w - (self.dw * lr)
    self.b = self.b - (self.db * lr)
    
class Dropout(BaseLayer):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p  # probability to drop

  def forward(self, x):
    # when in evaluation mode - skip dropout
    if not self.is_training:
        return x

    # create mask (keep = 1, drop = 0)
    self.mask = (np.random.rand(*x.shape) > self.p)

    # inverted dropout
    return x * self.mask / (1 - self.p)

  def backward(self, dout):
    # pass gradient only where neurons survived
    return dout * self.mask / (1 - self.p)