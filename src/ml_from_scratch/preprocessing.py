import numpy as np

class ZScoreNormalization:
  EPSILON = 1e-8

  def fit(self, x):
    self.mean = x.mean(axis=0)
    self.std = x.std(axis=0)

  def transform(self, x):
    if not hasattr(self, "mean") or not hasattr(self, "std"):
      raise ValueError("ZScoreNormalization must be fit before calling transform.")

    return (x - self.mean) / (self.std + self.EPSILON) # to account for zero std

  def fit_transform(self, x):
    self.fit(x)
    return self.transform(x)