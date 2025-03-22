# Toy Neural Architecture Search with E2B and Anthropic for MNIST Classification

What? 

Machine learning engineers create neural network architectures using tedious work. There is a approache like neural architecture search to iterate basic neural network building blocks in for loops to automate the process, which is limited that could be. We tried to combine LLM and E2B for the search.

How? 

We use LLM to design a neural network in pytorch for the toy problem  - MNIST classification (classify an image with a number to a correct class). Validate the model using E2B to make sure the training can be done. Train the model and pick the best architecture.

**We used E2B to validate the LLM provided valid Python code in Pytorch with the help of a custom sandbox.**

## Install

Make sure you have *Python 3.12* and latest *npm*.

### Setup API keys

Setup API keys for Anthropic and E2B.

### Install requirements

> pip install -r requirements

### Create a custom E2B sandbox

> npm i -g @e2b/cli
> e2b template build --name "torch_template" -c "/root/.jupyter/start-up.sh"

## Run

> python main.py

Generated neural network architectures are stored in the *generated_nn_architectures* dir and their training logs are stored in the *generated_nn_architectures_training* dir. 