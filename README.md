# Machine Learning from Scratch (NumPy)

This repository implements core machine learning and neural network components from scratch using only NumPy, with a focus on understanding both the mathematics and the engineering behind model training.

Each notebook builds on the previous one, covering forward and backward propagation, optimization, regularization, and evaluation. The goal is to bridge the gap between theory and practical implementation without relying on high-level ML frameworks.

---

## Key Concepts Covered

* Forward and backward propagation
* Gradient descent and backpropagation
* Loss functions and their derivatives
* Model evaluation and validation
* Cross-validation
* Regularization techniques such as L2 and Dropout
* Evaluation metrics such as Accuracy, Precision, Recall, and F1
* Threshold tuning for binary classification

All models are implemented from scratch, without using machine learning frameworks such as PyTorch or TensorFlow.

---

## Notebook progression

Each notebook builds on the previous one.

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

### 04 — Regularization (L2 & Dropout)

* Binary classification with sigmoid and binary cross-entropy
* L2 regularization
* Dropout regularization
* Overfitting demonstration
* Comparison of regularization strategies

### 05 — Evaluation Metrics

* Why accuracy alone is often misleading in real-world business problems
* Introduction to precision, recall, and F1 score
* Threshold tuning for binary classification and metric trade-offs

---

## Why this project exists

High-level frameworks make it easy to train models, but they also hide many of the details that matter when debugging, improving, and reasoning about learning systems.

This project focuses on understanding how models actually work internally by implementing their components manually.