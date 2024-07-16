# FakeNews-GNN

## Fraud Detection with Pytorch Geometric
This Colab notebook demonstrates how to perform fraud detection using Pytorch Geometric on the UPFD dataset. The UPFD dataset contains news propagation graphs extracted from Twitter, and the goal is to classify news articles as real or fake.

## Installation
The notebook requires PyTorch and PyTorch Geometric to be installed. The following code snippet installs these libraries:
```
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install -q git+https://github.com/rusty1s/pytorch_geometric.git !pip install networkx
```

## Dataset
The notebook uses the UPFD dataset, which can be downloaded and preprocessed using the following code:

```
from torch_geometric.datasets import UPFD
train_data = UPFD(root=".", name="gossipcop", feature="content", split="train")
test_data = UPFD(root=".", name="gossipcop", feature="content", split="test")
```

## Model and Training

The notebook defines a Graph Neural Network (GNN) model using GATConv layers for graph convolutions and a linear layer for readout. The model is trained using the Adam optimizer and binary cross-entropy loss.

## Evaluation

The notebook evaluates the trained model on the test set using accuracy and F1-score as metrics.

## Usage

To run the notebook, simply open it in Google Colab and execute the cells. The notebook will download the UPFD dataset, train the GNN model, and evaluate its performance.
