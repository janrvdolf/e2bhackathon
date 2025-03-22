# You can use most Debian-based base images
#FROM ubuntu:22.04

# Install dependencies and customize sandbox
# Make sure to use this base image
FROM e2bdev/code-interpreter:latest

# Install some Python packages
RUN pip install torch

