# Toy Neural Architecture Search with E2B and Anthropic for MNIST Classification

### What? 

Machine learning engineers typically design neural network architectures through tedious manual effort. Approaches such as Neural Architecture Search automate this process by iteratively evaluating basic neural network building blocks, but these methods have limitations. We attempted to combine LLM and E2B to enhance this search process.

### How? 

We use an LLM to design a neural network in PyTorch for a toy problem—MNIST classification (classifying an image containing a digit into the correct class). We validate the model using E2B to ensure that training can be successfully completed. Finally, we train the models and select the best-performing architecture.

**We used E2B along with a custom sandbox to validate that the LLM provided correct Python code in PyTorch.**

### Next Next

Add your custom dataset, make custom features search and use deep search with the latest code and papers available. You will get the best models #noFOMO.

## Install

Make sure you have *Python 3.12* and the latest *npm*.

### Setup API keys

Setup API keys for Anthropic and E2B.

### Install requirements

> pip install -r requirements

### Create a custom E2B sandbox

> npm i -g @e2b/cli

> e2b template build --name "torch_template" -c "/root/.jupyter/start-up.sh"

## Run

> python main.py

There are three stages: 
 - neural network generation using LLM and E2B
 - training
 - evaluation by LLM to decide the best architecture trained from generated architectures

Generated neural network architectures are stored in the *generated_nn_architectures* dir and their training logs are stored in the *generated_nn_architectures_training* dir. 