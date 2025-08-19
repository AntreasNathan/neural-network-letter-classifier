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

## Instructions to run the Neural Network:
1) You need to have in the same folder as Ex2_Final.py the parameters.txt,train.txt and test.txt file
2) Open terminal and navigate to the file referred in step 1
3) Execute the program with the command 'python Ex2_Final.py'
Note: The program uses the numpy, matplotlib and random libraries, in case there are errors about that, the libraries can be install with the command: 'pip install numpy'(the same goes for the other libraries)(run these commands in terminal before running the program)'
4) Results should be printed in files errors.txt, successrate.txt
