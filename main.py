"""
This script is used to generate neural network architectures using the Anthropic model, validate them in E2B and then train them.
"""

import argparse
import os
from anthropic import Anthropic
from e2b_code_interpreter import Sandbox


args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    "--number_of_nn_architectures",
    default=3,
    type=int,
    help="Number of neural networks you want to generate.",
)
args_parser.add_argument(
    "--output_dir",
    default="generated_nn_architectures",
    type=str,
    help="Dir path where are generated neural architectures stored.",
)
args_parser.add_argument(
    "--output_dir_training",
    default="generated_nn_architectures_training",
    type=str,
    help="Dir path where are generated neural architectures stored.",
)
args = args_parser.parse_args()

# Neural architecture search

client = Anthropic()
system_prompt = "You are a helpful assistant that can execute python code in a safe environment. Only respond with the code to be executed and nothing else. Strip backticks in code blocks."
prompt = """
You are given the following code snippet. Implement the missing part of the model. The model is classifing MNIST dataset. 
Make sure the model is executable. Replace only the TODO part. Return the executable Python code:


import torch 

model = torch.nn.Sequential(
        *[
            <TODO: implement>
        ]
    )

"""


for i in range(args.number_of_nn_architectures):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract code from response
    code = response.content[0].text

    if code:
        with Sandbox("torch_template") as sandbox:
            execution = sandbox.run_code(code)

            if not execution.error:
                result = execution.text
                print("Execution run: ", i, "is successful")

                # Write file to local filesystem
                with open(
                    os.path.join(
                        args.output_dir,
                        "__output_code_" + str(i) + ".py",
                    ),
                    "w",
                ) as file:
                    print(
                        f'Writing candidate neural network to a file "__output_code_{i}.py"...'
                    )
                    file.write(code)
            else:
                print("Execution run: ", i, "has an error")
                print(execution.error)

# Train neural networks

for i in range(args.number_of_nn_architectures):
    os.system(
        " ".join(
            [
                "python",
                "mnist_training.py",
                "--generated_nn_architecture",
                f"generated_nn_architectures.__output_code_{i}",
                f">{args.output_dir_training}/train{i}.txt",
            ]
        )
    )

# Pick the best neural network after training

system_prompt = (
    "You are a helpful assistant that is also an expert in neural networks." ""
)
prompt = f"You are given {args.number_of_nn_architectures} neural network architectures trained on MNIST dataset. Choose the best one and explain why."

train_outputs_prompts = []
for i in range(args.number_of_nn_architectures):
    with open(
        os.path.join(
            f"{args.output_dir_trainin}/train{i}.txt",
        ),
        "r",
    ) as file:
        file_content = file.read()
        train_outputs_prompts.append(
            f"Train log for neural network {i}:\n{file_content}"
        )

response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": prompt},
        *[
            {"role": "user", "content": train_outputs_prompt}
            for train_outputs_prompt in train_outputs_prompts
        ],
    ],
)

# Extract code from response
explanation = response.content[0].text

print("The best neural network: ", explanation)
