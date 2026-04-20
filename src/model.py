import numpy as np
from .layers import LinearLayer

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

  # new train and eval functions to update layers
  def train(self):
    for layer in self.layers:
      layer.train()

  def eval(self):
    for layer in self.layers:
      layer.eval()

  def save(self, filepath):
    params = {}
    for i, layer in enumerate(self.layers):
      if isinstance(layer, LinearLayer):
        params[f'layer_{i}_w'] = layer.w
        params[f'layer_{i}_b'] = layer.b
    np.savez(filepath, **params)
    print(f"[Info] Model parameters saved to {filepath}")

  def load(self, filepath):
    loaded_params = np.load(filepath)
    for i, layer in enumerate(self.layers):
      if isinstance(layer, LinearLayer):
        if f'layer_{i}_w' in loaded_params and f'layer_{i}_b' in loaded_params:
          layer.w = loaded_params[f'layer_{i}_w']
          layer.b = loaded_params[f'layer_{i}_b']
        else:
          print(f"[Warning] Parameters for layer {i} not found in {filepath}")
    print(f"[Info] Model parameters loaded from {filepath}")
