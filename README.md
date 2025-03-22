# Toy Neural Architecture Search with E2B and Anthropic for MNIST Classification

What? 

Machine learning engineers create neural networks architectures that are hard to find. There is a field of auto ML and approaches like neural architecture search to iterate basic neural network building blocks in for loops.

How? 

We use LLM to design a neural network in pytorch for a toy problem like MNIST classification (classify an image with a number to a correct class). Validate the model using E2B to make sure the training can be done. Train the model and pick the best architecture.

We used E2B to validate the LLM provided valid Python code in Pytorch with the help of a custom sandbox.

## Install

### Setup API keys

Setup API keys for Anthropic and E2B.

### Install requirements

> pip install -r requirements

### E2B Sandbox

Create a custon E2B sandbox.

## Run

> python main.py

Generated neural network architectures are stored in *generated_nn_architectures* dir and their training logs are stored in *generated_nn_architectures_training*. 