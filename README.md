# Linear Regression From Scratch

A simple implementation of **Linear Regression using Gradient Descent**, built from scratch with Python and NumPy.

## Overview

This project was created to understand the fundamentals behind linear regression and gradient descent rather than relying on machine learning libraries such as Scikit-learn.

The model learns two parameters:

* **w** — weight
* **b** — bias

The model uses them to make predictions:

[
f(x) = wx + b
]

## What I Implemented

* Linear Regression model
* Cost function
* Gradient calculation
* Gradient Descent
* Model training
* Predictions on new data

The goal is to minimize the cost function and improve the model's predictions.

## Example Results

Starting with:

```text
w = 0
b = 0
```

After 1000 iterations:

```text
Final w ≈ 0.2205
Final b ≈ 0.1317
```

The cost decreased from:

```text
Initial Cost: 49518.0
Final Cost:   13579.19
```

The trained model can then be used to make predictions on new inputs.

## Technologies

* Python
* NumPy

## Purpose

This project is part of my journey in learning **Machine Learning**, with a focus on understanding the mathematics and algorithms behind common ML models before using higher-level libraries.
