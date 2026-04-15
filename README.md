# Machine Learning from Scratch (NumPy)

As I learn more about machine learning, my curiosity would not let me continue without developing a deeper understanding of how neural networks work beneath the framework wrappers available today. Even though LLMs can generate models, training loops, and in some cases even synthetic training data, I believe that for personal development it is best to go beyond a surface-level understanding when implementing a neural network.

This repository demonstrates the mathematical foundations of machine learning by implementing core algorithms from scratch using only NumPy.

---

## Contents

### 01 — Linear Regression & Backpropagation

* Linear regression from scratch
* Mean Squared Error (MSE)
* Gradient descent
* Manual derivation of gradients
* Backpropagation through a simple neural network
* Implementation of `LinearLayer` and `ReLU`

### 02 — Softmax & Cross Entropy

* Softmax activation (numerically stable)
* Cross-entropy loss
* One-hot encoding
* Multi-class classification (Iris dataset)
* Mini-batch training

### 03 — K-Fold Cross-Validation

* Train/validation split vs cross-validation
* K-fold cross-validation from scratch
* Data leakage and proper preprocessing
* Model evaluation across multiple folds

---

## Key Concepts Covered

* Forward pass and backward pass
* Chain rule in practice
* Gradient flow through layers
* Loss functions and their derivatives
* Basic model training
* Model evaluation and validation strategies
* Cross-validation and variance estimation
* Data leakage prevention

---

## Goals

As mentioned above, I created this project as part of my learning journey with the goal of satisfying my curiosity. My goals are:

* Understand how machine learning models work internally
* Build intuition for gradients and optimization
* Bridge the gap between theory and implementation
* Go deeper than what most tutorials and courses show and explain

All models are implemented from scratch, without using machine learning frameworks such as PyTorch or TensorFlow.