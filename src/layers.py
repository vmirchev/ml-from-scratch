from abc import ABC, abstractmethod
import numpy as np

class BaseLayer(ABC):
  is_training: bool

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
  def __init__(self, in_features:int, out_features:int, initialization: str = "he"):

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

  def forward(self, x):
    self.x = x
    return x @ self.w + self.b

  def backward(self, dout):
    self.dw = self.x.T @ dout
    self.db = np.sum(dout, axis=0, keepdims=True)
    dx = dout @ self.w.T
    return dx

  def step(self, lr):
    self.w = self.w - (self.dw * lr)
    self.b = self.b - (self.db * lr)