# Continual Growing

A Bayesian neural network that grows wider as it learns new tasks sequentially, using weight uncertainty to protect previously learned knowledge from being overwritten.

## Background

Standard neural networks suffer from **catastrophic forgetting**: training on a new task overwrites weights learned for previous tasks. This project addresses this with two mechanisms:

1. **Uncertainty-scaled learning rates** — each weight's learning rate is proportional to its own standard deviation (σ). Weights that became confident on past tasks get very small updates; uncertain weights stay plastic.
2. **Dynamic growth** — when newly added neurons saturate (σ drops below a threshold), the network grows wider by appending fresh neurons. Old weights are untouched; new neurons learn freely.

Networks are Bayesian: each weight is a distribution parameterized by a mean `μ` and `ρ`, where `σ = softplus(ρ)`. Training minimizes the ELBO (Evidence Lower Bound) — a combination of KL divergence from the prior and negative log-likelihood.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd continual-growing
uv sync
```

## Quickstart

All scripts must be run from the `src/` directory:

```bash
cd src

# Growing network on split-MNIST (5 binary tasks)
python run.py --experiment mnist5 --train_mode grow --device cpu --wandb_mode offline

# Fixed-size baseline (no growth) for comparison
python run.py --experiment mnist5 --train_mode grow --growth_rate 0 --device cpu --wandb_mode offline
```

At the end of each run, the console prints an accuracy matrix and summary metrics:
- **ACC** — average accuracy across all tasks after the final task
- **BWT** — backward transfer; negative values indicate forgetting

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--experiment` | *(required)* | `mnist2`, `mnist5`, `pmnist`, `cifar`, `mixture` |
| `--cl_mode` | `domain-incremental` | `task-incremental` (separate head/task) or `domain-incremental` (shared head) |
| `--device` | `mps` | `cpu`, `cuda`, or `mps` |
| `--wandb_mode` | `online` | `online` or `offline` |
| `--hidden_n` | `16` | Starting number of neurons per hidden layer |
| `--epochs` | `10` | Epochs per task |
| `--growth_rate` | `5` | Neurons added per growth event (0 = disabled) |
| `--growth_saturation` | `0.2` | Fraction of new params that must saturate to trigger growth |
| `--growth_threshold` | `0.05` | σ below which a parameter is considered saturated |
| `--regularization` | `sns` | Prior type: `bbb` (scale mixture), `unimodal` (single Gaussian), `sns` (spike & slab on σ) |
| `--static` | — | Disable Bayesian sampling; use only means (point estimate) |

Full argument list in `src/run.py`.

## Continual Learning Modes

**Task-incremental**: at test time the task identity is known. Each task gets its own output head (`classifier[0]`, `classifier[1]`, …). Easier, because the model only has to be right within each task's class set.

**Domain-incremental**: task identity is unknown at test time. A single shared output head is used for all tasks. Harder and more realistic.

## Repository Structure

```
src/
  run.py               # Entry point: argument parsing, data loading, training loop
  utils.py             # Logging, directory creation, BWT/FWT metrics

  train/
    trainer.py         # Trainer class: growth logic, uncertainty-scaled LR, ELBO loss
    utils.py           # BayesianSGD optimizer (supports per-parameter tensor LRs)

  networks/
    FC.py              # BayesianLinear: core layer with grow_output / grow_input
    mlp_grow.py        # BayesianMLP: 1–2 hidden layers, growth-enabled
    resnet_grow.py     # BayesianResNet: used for cifar/mixture (growth not yet implemented)
    distributions.py   # VariationalPosterior, Prior variants
    BayesianConvs.py   # Bayesian conv layers (used by ResNet)
    BatchNorm.py       # Bayesian batch norm

  dataloaders/
    mnist2.py          # MNIST split into 2 tasks (digits 0–4 vs. 5–9)
    mnist5.py          # MNIST split into 5 binary tasks (one pair of digits each)
    pmnist.py          # Permuted MNIST: same digits, different pixel permutation per task
    cifar.py           # CIFAR split into tasks
    mixture.py         # Mixed dataset tasks

data/          # Downloaded automatically on first run
checkpoints/   # Model checkpoints saved after each task
```

## Experiment Tracking

Runs are logged to [Weights & Biases](https://wandb.ai) under the project `continual_growing`. Pass `--wandb_mode offline` to skip network upload and log locally only.
