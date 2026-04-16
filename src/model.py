import numpy as np

class Model:
  def __init__(self, layers=None):
    self.layers = []
    if layers:
        self.add_all(layers)

  def add(self, layer):
    self.layers.append(layer)

  def add_all(self, layers):
    self.layers.extend(layers)

  def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x

  def backward(self, grad):
    # iterate backward through the layers
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return grad

  def step(self, lr):
    # call step only on layers that have learnable parameters
    for layer in self.layers:
        if hasattr(layer, 'step'):
            layer.step(lr)