# Assignment 1: Multi-Layer Perceptron for Image Classification

**Soumya Ranjan Patel · DA25M029** 

Github link - 
W&B Report Link - https://api.wandb.ai/links/samwellthorson04-iit-madras/my889unx

## Overview

This project implements a fully configurable **Multi-Layer Perceptron (MLP)** from scratch using **NumPy**. The model supports multiple optimization algorithms, activation functions, loss functions, and weight initialization methods. The implementation is designed to train and evaluate models on the **MNIST** and **Fashion-MNIST** datasets.

The project follows the structure required by the DA6401 assignment and includes experiment tracking using **Weights & Biases (W&B)**.

---

## Features

* Fully modular neural network implementation
* Forward and backward propagation implemented from scratch
* Multiple optimizers
* Multiple activation functions
* Configurable network depth and width
* L2 weight decay support
* Gradient logging for analysis
* W&B experiment tracking and hyperparameter sweeps
* CLI-based configuration
* Model serialization and inference pipeline

---

## Supported Components

### Optimizers

* SGD
* Momentum
* NAG
* RMSProp

### Activation Functions

* ReLU
* Tanh
* Sigmoid

### Loss Functions

* Cross Entropy
* Mean Squared Error

### Weight Initialization

* Random
* Xavier

---

## Dataset

The following datasets are supported:

* **MNIST**
* **Fashion-MNIST**

Datasets are loaded using `keras.datasets`.

Images are normalized and flattened before being passed to the network.

---

## Project Structure

```
da6401_assignment_1/
│
├── src/
│   ├── train.py
│   ├── inference.py
│   │
│   ├── ann/
│   │   ├── neural_network.py
│   │   ├── neural_layer.py
│   │   ├── optimizers.py
│   │   ├── objective_functions.py
│   │   ├── activations.py

│   ├── utils/
│       ├── data_loader.py
│ 
│   ├── best_model.npy
│   ├── best_config.json
│
├── sweep.yaml
│
└── README.md
```

---

## Installation

Create a virtual environment and install dependencies.

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy scikit-learn matplotlib wandb tensorflow
```

---

## Training

Run training using the CLI interface.

Example:

```bash
python train.py \
-d mnist \
-e 20 \
-b 64 \
-o rmsprop \
-lr 0.001 \
-nhl 3 \
-sz 128 128 64 \
-a relu \
-w_i xavier \
-w_p da6401_assignment1 \
--model_save_path best_model.npy
```

---

## Command Line Arguments

| Argument | Description                        |
| -------- | ---------------------------------- |
| `-d`                | Dataset (`mnist`, `fashion_mnist`) |
| `-e`                | Number of training epochs          |
| `-b`                | Batch size                         |
| `-o`                | Optimizer                          |
| `-lr`               | Learning rate                      |
| `-wd`               | Weight decay                       |
| `-nhl`              | Number of hidden layers            |
| `-sz`               | Hidden layer sizes                 |
| `-a`                | Activation function                |
| `-l`                | Loss function                      |
| `-w_i`              | Weight initialization              |
| `-w_p`              | W&B Project                        |
| `--model_save_path` | Weight initialization              |
---

## Inference

Evaluate a trained model using:

```bash
python inference.py \
  -d fashion_mnist \
  -nhl 3 \
  -sz 128 128 128 \
  -a relu \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0001 \
  -l cross_entropy \
  -w_i xavier \
  --model_path best_model.npy
```

Metrics reported:

* Accuracy
* Precision
* Recall
* F1 Score

---


## Hyperparameter Sweep

W&B sweeps are used to explore hyperparameters.

Run:

```bash
wandb sweep sweep.yaml
wandb agent <SWEEP_ID> --count 100
```

This performs a grid search across:

* optimizers
* learning rates
* activations
* architectures

---

## Experiments Conducted

The following experiments were performed as part of the W&B report:

1. Data exploration and visualization
2. Hyperparameter sweep (100 runs)
3. Optimizer convergence comparison
4. Vanishing gradient analysis
5. Dead neuron investigation
6. Loss function comparison
7. Global performance analysis
8. Error analysis using confusion matrices
9. Weight initialization symmetry experiment
10. Fashion-MNIST transfer experiment

---

## Results

Best configuration discovered during sweep:

```
Optimizer: SGD
Learning Rate: 0.1
Activation: ReLU
Architecture: [128,128,128]
Weight Initialization: Xavier
Batch Size: 64
loss: cross entropy
hidden layers: 3
weight decay: 0.0001
weight initialization: Xavier
```

Performance on MNIST:

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~98%  |
| F1 Score | ~0.97 |

---



Soumya Ranjan Patel
DA6401 – Deep Learning
