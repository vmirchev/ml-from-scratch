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

  def save(self, filepath, log_info: bool = False):
    params = {}
    for i, layer in enumerate(self.layers):
      if isinstance(layer, LinearLayer):
        params[f'layer_{i}_w'] = layer.w
        params[f'layer_{i}_b'] = layer.b
    np.savez(filepath, **params)

    if log_info:
      print(f"[Info] Model parameters saved to {filepath}")

  def load(self, filepath, strict: bool = True, log_info: bool = False):
    loaded_params = np.load(filepath)
    
    expected_keys = set()
    loaded_keys = set()

    for i, layer in enumerate(self.layers):
      if not isinstance(layer, LinearLayer):
        continue

      w_key = f"layer_{i}_w"
      b_key = f"layer_{i}_b"

      expected_keys.update([w_key, b_key])

      if w_key not in loaded_params or b_key not in loaded_params:
        if strict:
          raise ValueError(
              f"Missing parameters for layer {i}: "
              f"{w_key if w_key not in loaded_params else ''} "
              f"{b_key if b_key not in loaded_params else ''}"
          )
        elif log_info:
          print(f"[Warning] Missing params for layer {i}")
        continue

      w = loaded_params[w_key]
      b = loaded_params[b_key]

      # shape validation
      if layer.w.shape != w.shape:
        raise ValueError(f"Shape mismatch for {w_key}: expected {layer.w.shape}, got {w.shape}")

      if layer.b.shape != b.shape:
        raise ValueError(f"Shape mismatch for {b_key}: expected {layer.b.shape}, got {b.shape}")

      layer.w = w
      layer.b = b

      loaded_keys.update([w_key, b_key])

      if log_info:
        print(f"[Info] Loaded layer {i}: w{w.shape}, b{b.shape}")

    # extra keys check
    unexpected_keys = set(loaded_params.files) - expected_keys
    if strict and unexpected_keys:
      raise ValueError(f"Unexpected keys in file: {unexpected_keys}")

    # missing keys check
    if strict:
      missing = expected_keys - loaded_keys
      if missing:
        raise ValueError(f"Incomplete model load. Missing keys: {missing}")

    if log_info:
      print(f"[Info] Model parameters loaded from {filepath}")
