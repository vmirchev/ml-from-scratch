import numpy as np

class LinearLayer:
    def __init__(self, in_features:int, out_features:int):
        self.w = 0.01 * np.random.randn(in_features, out_features) # make weights really small
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
