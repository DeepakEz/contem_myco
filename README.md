# Contemplative Multi-Agent Reinforcement Learning

**Research code for ICLR submission: Integrating Ethical Constraints, Stigmergic Communication, and Mindfulness into Multi-Agent Systems**

## Abstract

This repository contains the implementation for our research on contemplative multi-agent reinforcement learning (MARL). We introduce three novel modules that enhance multi-agent coordination:

1. **Ethical Constraints Module** - Lagrangian-based optimization for harm avoidance, fairness, and cooperation
2. **Stigmergic Diffusion Module** - Mycelium-inspired indirect communication through environmental signals
3. **Mindfulness Module** - Uncertainty-aware decision making with ensemble predictions and action smoothing

We evaluate against established MARL baselines (CommNet, TarMAC, QMIX) on mixed-motive environments requiring both cooperation and individual performance.

## Key Contributions

### Module A: Ethical Constraints in MARL

We formulate ethical behavior as constrained optimization using Lagrangian methods:

```
max_θ E[R(θ)] s.t. C_harm ≤ ε_h, C_fairness ≤ ε_f, C_coop ≥ τ_c
```

Components:
- **Multi-framework ethics evaluation**: Neural network scoring actions across consequentialist, deontological, virtue, and care ethics frameworks
- **Lagrangian optimizer**: Dual gradient ascent for constraint satisfaction
- **Fairness via Gini coefficient**: Measuring reward inequality across agents

### Module B: Stigmergic Diffusion Communication

Inspired by fungal mycelium networks, agents communicate indirectly through environmental signal fields:

- **Diffusion field**: 2D grid with deposit, sense, diffuse, decay dynamics
- **Multi-channel signals**: Multiple signal types for different information
- **Emergent coordination**: Agents learn to encode/decode field patterns

### Module C: Mindfulness for Robustness

Ensemble-based uncertainty estimation enables adaptive behavior:

- **Ensemble predictor**: Multiple forward models estimate state uncertainty
- **Gating mechanism**: Switches between reactive/conservative policies based on surprise
- **Action smoothing**: Temporal smoothing reduces oscillation under uncertainty

## Installation

```bash
# Clone repository
git clone https://github.com/DeepakEz/contem_myco.git
cd contem_myco

# Install dependencies
pip install torch numpy pettingzoo tensorboard

# Optional: For full MPE environments
pip install pettingzoo[mpe]
```

## Quick Start

### Run Full Comparison (Recommended)

```bash
# Run all ablations + baselines with 30 seeds (ICLR-grade)
python -m research.run_experiment --full-comparison --seeds 30 --timesteps 1000000

# Quick test run (5 seeds, fewer timesteps)
python -m research.run_experiment --full-comparison --seeds 5 --timesteps 100000
```

### Run Specific Experiments

```bash
# Ablation study only
python -m research.run_experiment --ablation --seeds 30

# Baselines only (CommNet, TarMAC, QMIX)
python -m research.run_experiment --baselines --seeds 30

# Scaling experiments
python -m research.run_experiment --scaling --agents 4 8 16 32 --seeds 10

# Robustness tests (noise injection)
python -m research.run_experiment --robustness --seeds 20
```

### Using GPU

```bash
python -m research.run_experiment --full-comparison --seeds 30 --device cuda
```

## Experiment Configurations

### Ablation Conditions

| Configuration | Ethics | Diffusion | Mindfulness |
|--------------|--------|-----------|-------------|
| `contemplative_full` | Yes | Yes | Yes |
| `no_ethics` | No | Yes | Yes |
| `no_diffusion` | Yes | No | Yes |
| `no_mindfulness` | Yes | Yes | No |
| `no_modules` | No | No | No |

### Baselines

| Baseline | Description | Reference |
|----------|-------------|-----------|
| **CommNet** | Averaged hidden state communication | Sukhbaatar et al., 2016 |
| **TarMAC** | Attention-based targeted communication | Das et al., 2019 |
| **QMIX** | Value decomposition with monotonic mixing | Rashid et al., 2018 |

## Project Structure

```
research/
├── __init__.py
├── config.py                    # Experiment configurations
├── run_experiment.py            # Main experiment runner
│
├── agents/
│   ├── __init__.py
│   ├── contemplative_agent.py   # Main agent with all modules
│   │
│   ├── modules/
│   │   ├── ethics.py            # Module A: Ethical constraints
│   │   ├── diffusion.py         # Module B: Stigmergic diffusion
│   │   └── mindfulness.py       # Module C: Mindfulness
│   │
│   └── baselines/
│       ├── commnet.py           # CommNet baseline
│       ├── tarmac.py            # TarMAC baseline
│       └── qmix.py              # QMIX baseline
│
├── environments/
│   └── mpe_wrapper.py           # Mixed-motive MPE wrapper
│
└── training/
    └── mappo.py                 # MAPPO trainer with GAE
```

## Output Format

Results are saved to `experiments/<timestamp>/`:

```
experiments/20260129_143022/
├── aggregate_results.json       # Summary statistics
├── baselines/
│   ├── commnet/
│   │   └── seed_*/results.json
│   ├── tarmac/
│   │   └── seed_*/results.json
│   └── qmix/
│       └── seed_*/results.json
└── contemplative/
    ├── contemplative_full/
    │   └── seed_*/
    │       ├── results.json
    │       ├── checkpoints/
    │       └── tensorboard/
    ├── no_ethics/
    ├── no_diffusion/
    ├── no_mindfulness/
    └── no_modules/
```

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average episode reward |
| `social_welfare` | Sum of all agent rewards |
| `gini_coefficient` | Reward inequality (0=equal, 1=unequal) |
| `cooperation_rate` | Fraction of cooperative actions |
| `ethics_score` | Average ethical evaluation score |
| `constraint_violations` | Number of ethical constraint breaches |

## Expected Results

Based on preliminary experiments, we expect:

1. **Full contemplative agent** outperforms no-module baseline by 15-25% on social welfare
2. **Ethics module** reduces Gini coefficient by ~0.1 (more equitable rewards)
3. **Diffusion module** improves coordination in sparse reward settings
4. **Mindfulness module** provides robustness under observation noise
5. **CommNet/TarMAC** competitive on reward but lower on fairness metrics
6. **QMIX** strong individual performance but may sacrifice cooperation

## Reproducibility

All experiments use:
- 30 random seeds for statistical significance
- Fixed hyperparameters (see `research/config.py`)
- Deterministic PyTorch operations where possible

Set seeds explicitly:
```python
import torch
import numpy as np
torch.manual_seed(seed)
np.random.seed(seed)
```

## TensorBoard Visualization

```bash
tensorboard --logdir experiments/
```

View:
- Training curves (reward, loss)
- Ethical constraint satisfaction
- Communication patterns (attention weights for TarMAC)

## Citation

```bibtex
@inproceedings{contemplative_marl_2026,
  title={Contemplative Multi-Agent Reinforcement Learning:
         Integrating Ethics, Stigmergy, and Mindfulness},
  author={},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## References

- Sukhbaatar, S., Szlam, A., & Fergus, R. (2016). Learning multiagent communication with backpropagation. NeurIPS.
- Das, A., Gerber, T., Sabach, S., & Kottur, S. (2019). TarMAC: Targeted multi-agent communication. ICML.
- Rashid, T., Samvelyan, M., Schroeder, C., et al. (2018). QMIX: Monotonic value function factorisation. ICML.
- Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). Proximal policy optimization algorithms. arXiv.

## License

MIT License - see LICENSE file.

---

**Note**: This is research code accompanying an academic paper. For production use, additional testing and optimization would be required.
