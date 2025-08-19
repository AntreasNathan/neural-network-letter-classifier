# Neural Network for Handwritten Letter Recognition

This project implements a **Feedforward Neural Network** with **Backpropagation** and **Gradient Descent** to recognize pre-processed handwritten letters.  
Developed in Python as part of a university machine learning assignment.

---

## Project Overview
- The neural network uses a **sigmoid activation function** for outputs.  
- Training and testing are performed on pre-processed datasets (`train.txt` and `test.txt`).  
- To improve efficiency, the original loop-heavy implementation was refactored into **vectorized operations** using NumPy.  
- The dataset contains 16 normalized features (0–1) representing each letter. Each sample is converted into a **26-dimensional target vector** (one-hot encoding for A–Z).  
- The model predicts the letter corresponding to the highest output value.  
- Data shuffling is applied to prevent memorization and ensure general learning.  
- Training set: **70%** of data  
- Testing set: **30%** of data  
