# EchoStateNetwork

## Description

EchoStateNetwork is an implementation of an Echo State Network (ESN) by Defne. It includes a validation method for automatic hyperparameter tuning and a demonstration on the Lorenz-63 dataset.

## Purpose

The main purpose of this project is to provide a reference implementation of an Echo State Network for the MagriLab. It serves as a starting point for understanding and working with ESNs in the context of chaotic dynamical systems.

## Installation

Requires Python 3.10 or newer.

**Recommended (uv):**

```bash
uv sync --extra dev
```

**pip:**

```bash
pip install -e ".[dev]"
```

**pip with requirements file:**

```bash
pip install -r requirements.txt
```

Run the Lorenz-63 tutorial from the repository root so `esn` and `jax_esn` imports resolve:

```bash
jupyter notebook tutorial/lorenz63_esn_demo.ipynb
```

### Private PyPI index

If your machine must use a corporate package index, configure it locally (do not commit index URLs). With uv, either:

- set `UV_INDEX_URL` in your environment, or
- copy `uv.local.toml.example` to `uv.local.toml` and adjust the index URL.

Reach out to Defne or Elise in case of questions!
