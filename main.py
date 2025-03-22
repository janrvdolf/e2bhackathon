"""
This script is used to generate neural network architectures using the Anthropic model, validate them in E2B and then train them.
"""

import argparse
import os
import json
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

print("🤖 Generate neural network architectures using E2B...")

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
                print("E2B execution run: ", i, "is successful - saving...")

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

                print("Generated neural network architecture:")
                print(code)
                print("\n")
            else:
                print("E2B execution run: ", i, "has an error - skipping...")
                print(execution.error)

# Train neural networks

print("\n🤖 Train neural networks...")

for i in range(args.number_of_nn_architectures):
    print(f"Neural network {i} (wait a minute):")
    training_log_path = f"{args.output_dir_training}/train{i}.txt"
    os.system(
        " ".join(
            [
                "python",
                "mnist_training.py",
                "--generated_nn_architecture",
                f"generated_nn_architectures.__output_code_{i}",
                f">{training_log_path}",
            ]
        )
    )
    with open(training_log_path, "r") as file:
        print(file.read())

# Pick the best trained neural network

print("\n🤖 Picking the best neural network...\n")

system_prompt = "You are a helpful assistant and an expert in neural networks." ""
prompt = f'You are given {args.number_of_nn_architectures} neural network architectures trained on MNIST dataset. Choose the best one and explain why. Return JSON {{"best_network": <int>., "explanation": <str>}}.'

train_outputs_prompts = []
for i in range(args.number_of_nn_architectures):
    with open(
        os.path.join(
            f"{args.output_dir_training}/train{i}.txt",
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

response_json = json.loads(response.content[0].text)


print("🏆 The best network: ")
print("Best network: ", response_json["best_network"])
print("Explanation: ", response_json["explanation"])

print("\n Architecture:")
print(
    open(
        os.path.join(
            args.output_dir,
            f"__output_code_{response_json['best_network']}.py",
        ),
        "r",
    ).read()
)


print("\n Training run:")
print(
    open(
        os.path.join(
            args.output_dir_training,
            f"train{response_json['best_network']}.txt",
        ),
        "r",
    ).read()
)

# Extract code from response
explanation = response.content[0].text

with open("best_neural_network.txt", "w") as f:
    f.write(explanation)
