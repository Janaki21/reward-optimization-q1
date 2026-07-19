# Reproducibility Guide

## Scope

This repository contains controlled experiments examining reward optimization,
constrained reinforcement learning, and execution-time veto enforcement under
reward misspecification.

The controlled RLHF-style experiment is a neural response-selection analogue.
It is not a production-scale language-model experiment.

## Supported environment

The neural experiments were developed with:

- Python 3.10
- OmniSafe 0.5.0
- Safety-Gymnasium 0.4.1
- Gymnasium 0.28.1
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib

## Create the neural environment

```bash
conda create -n reward-neural python=3.10 -y
conda activate reward-neural
python -m pip install --upgrade pip
python -m pip install -r requirements-neural-ci.txt