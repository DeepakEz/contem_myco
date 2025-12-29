# Contemplative MycoNet++

Enhanced MycoNet++ with contemplative AI capabilities, featuring mindfulness, wisdom propagation, ethical reasoning, and collective meditation.

## 🚀 Current Status

**Latest Version:** Phase III Full Stack Implementation (December 2025)

### Recent Major Updates

- ✅ **Phase III Contemplative Overmind** - Complete with WisdomArchive, ThresholdRegulator, ContemplativeScheduler, OvermindBus, and NetworkX visualization
- ✅ **Wisdom Signal System** - 6 signal types emitting from agents to environment grid
- ✅ **Agent API Fix** - Resolved critical API mismatch between main loop and ContemplativeNeuroAgent
- ✅ **Network Dharma Compiler** - Ethical directive generation from colony state
- ✅ **OpenAI Gym Environment** - RL training infrastructure for overmind policies
- ✅ **Diagnostic Logging** - Comprehensive debugging for agent behavior verification

### ⚠️ Important: Verify You Have the Latest Code

If you're experiencing issues (zero wisdom, frozen metrics), ensure you have the latest commits:

```bash
git pull origin claude/mycoagent-resilience-setup-01JPQq2DCngyxmZey6TVvqsJ
git log --oneline -1  # Should show: aeb8d69 Add explicit diagnostic logging...
```

## Quick Start

### 1. Setup
Run the setup script to check dependencies and create example configurations:
```bash
python setup_script.py
```

### 2. Run Experiments

#### Using the convenient run scripts:
```bash
# Linux/Mac
./run.sh minimal     # Quick test (5 agents, 100 steps)
./run.sh basic       # Basic experiment (10 agents, 300 steps)
./run.sh advanced    # Full study (20 agents, 1000 steps)

# Windows
run.bat minimal
run.bat basic
run.bat advanced
```

#### Using the main script directly:
```bash
# Use preset configurations
python myconet_contemplative_main.py --config basic --verbose

# Use custom configuration file
python myconet_contemplative_main.py --config-file configs/basic_test.json --verbose

# Command line overrides
python myconet_contemplative_main.py --config basic --max-steps 500 --population 15
```

### 3. View Results
Results are saved to the specified output directory (e.g., `results_basic/`):
- `*_results.json`: Complete simulation data and analysis
- `checkpoint_*.json`: Periodic simulation checkpoints

## Key Features

### Contemplative Agents (myconet_contemplative_entities.py)
- **10-Step Update Loop**: Complete agent lifecycle with observation → decision → action → learning → wisdom emission
- **Neural Decision-Making**: PyTorch-based brain with mindfulness, wisdom, and ethical modules
- **Mindfulness Monitoring**: Tracks attention coherence and awareness levels
- **Wisdom Memory**: Stores and retrieves insights with temporal decay
- **Ethical Reasoning**: Multi-framework moral evaluation (consequentialist, deontological, virtue ethics, Buddhist ethics)
- **Contemplative States**: Ordinary, mindful, deep contemplation, collective meditation
- **Action Distribution Tracking**: Monitors which actions agents choose over time

### Wisdom Signal Network (myconet_wisdom_signals.py)
- **6 Signal Types**:
  - `SUFFERING_ALERT`: Emitted when agent energy < 0.3 or health < 0.4
  - `ETHICAL_INSIGHT`: High ethical alignment + confidence
  - `COMPASSION_GRADIENT`: When helping other agents
  - `MEDITATION_SYNC`: High mindfulness state (> 0.7)
  - `WISDOM_BEACON`: During wisdom sharing
  - `CONTEMPLATIVE_DEPTH`: Deep contemplation states
- **Propagation**: Chemical-signal-like diffusion with decay and amplification
- **Grid Integration**: Agents emit to and receive from WisdomSignalGrid each step

### Phase III Contemplative Overmind (myconet_contemplative_overmind.py)
- **WisdomArchive**: Long-term insight storage with retrieval and pattern analysis
- **ThresholdRegulator**: Adaptive parameter tuning based on colony performance
- **ContemplativeScheduler**: Ritual execution system (6 ritual types: meditation, conflict resolution, wisdom circles, etc.)
- **OvermindBus**: Attention routing and priority-based intervention dispatch
- **OvermindVisualizer**: NetworkX-based wisdom flow visualization
- **Network Dharma Compiler**: Generates ethical directives from colony state (8 directive types)
- **Multi-Objective Rewards**: Survival (30%), efficiency (20%), wisdom (20%), compassion (15%), ethics (15%)

### RL Training Infrastructure (myconet_gym_env_contemplative.py)
- **OpenAI Gym Environment**: ContemplativeMycoNetGymEnv for training overmind policies
- **17-Dimensional Observation Space**: Population, energy, health, mindfulness, wisdom, ethics, signals, cooperation, conflicts
- **6 Overmind Actions**: NO_ACTION, TRIGGER_MEDITATION, PROPAGATE_INSIGHT, ADJUST_COMPASSION, INITIATE_SHARING, CRISIS_INTERVENTION
- **Multi-Objective Reward Function**: Balances survival, efficiency, wisdom generation, compassion, and ethical alignment

### Evolution & Learning
- **Contemplative Trait Evolution**: Mindfulness capacity, wisdom thresholds, ethical sensitivity
- **Collective Behavior Emergence**: Network-wide meditation, cooperative resource sharing
- **Wisdom Propagation**: Insights spread through the network like nutrients in fungal networks
- **Brain Learning**: Experience-based updates to neural decision-making policies

## Configuration

### Key Parameters

#### Contemplative Settings
- `enable_contemplative_processing`: Enable/disable contemplative features
- `wisdom_signal_strength`: Intensity of wisdom signal propagation
- `collective_meditation_threshold`: Threshold for triggering network meditation
- `ethical_reasoning_depth`: Complexity of ethical evaluation
- `compassion_sensitivity`: Responsiveness to suffering signals

#### Simulation Settings
- `environment_width/height`: World size
- `initial_population`: Starting number of agents
- `max_steps`: Simulation duration
- `enable_overmind`: Enable network-level AI coordination

## Analysis Metrics

### Population Metrics
- Survival rates, energy/health levels, generation diversity

### Contemplative Metrics
- Wisdom generation/propagation rates, mindfulness levels, collective harmony

### Ethical Metrics
- Ethical decision ratios, moral consistency, behavioral trends

### Network Metrics
- Signal diversity, network coherence, wisdom flow efficiency

## Example Configurations

### Minimal Test (configs/minimal_test.json)
- 5 agents, 15x15 environment, 100 steps
- No Overmind, basic contemplative features
- Quick functionality test

### Basic Experiment (configs/basic_test.json)
- 10 agents, 25x25 environment, 300 steps
- Overmind enabled, full contemplative features
- Standard research experiment

### Advanced Study (configs/advanced_study.json)
- 20 agents, 40x40 environment, 1000 steps
- Enhanced contemplative parameters
- Long-term behavioral evolution study

## Research Applications

### Studying Collective Intelligence
- How does wisdom propagate through agent networks?
- What triggers network-wide contemplative states?
- How do ethical behaviors emerge and spread?

### Human-AI Symbiosis Research
- Network integration of human and AI nodes
- Collaborative wisdom generation
- Mutual benefit optimization

### Beneficial AI Development
- Training AI systems with built-in ethical reasoning
- Network-level safety mechanisms
- Contemplative approach to AI alignment

## Dependencies

### Required
- Python 3.7+
- NumPy
- Standard library modules (json, pathlib, logging, etc.)

### Optional
- PyTorch (for neural network functionality)
- Matplotlib (for visualization)
- Seaborn (for advanced plotting)

## File Structure

```
contemplative_myconet/
├── myconet_contemplative_core.py          # Core contemplative processing (ContemplativeProcessor, WisdomMemory, EthicalReasoning)
├── myconet_wisdom_signals.py              # Wisdom signal propagation system (WisdomSignalGrid, 6 signal types)
├── myconet_contemplative_brains.py        # Enhanced neural architectures (ContemplativeBrain, PyTorch modules)
├── myconet_contemplative_entities.py      # Contemplative agent classes (ContemplativeNeuroAgent with 10-step update)
├── myconet_contemplative_overmind.py      # Phase III overmind (WisdomArchive, Scheduler, Thresholds, Bus, Visualizer)
├── myconet_contemplative_config.py        # Unified configuration (ContemplativeConfig dataclass)
├── myconet_dharma_compiler.py             # Network dharma compiler (8 ethical directive types)
├── myconet_gym_env_contemplative.py       # OpenAI Gym environment for RL training
├── myconet_contemplative_main.py          # Main simulation script (ContemplativeSimulation)
├── setup_contemplative_myconet.py         # Setup and configuration helper
├── configs/                               # Configuration files
├── contemplative_results/                 # Simulation outputs (checkpoints, results, visualizations)
└── README.md                              # This file
```

## Troubleshooting

### Zero Wisdom / Frozen Metrics

If you see output like this:
```
Average Age: 0.0
Total Wisdom Generated: 0.0
Network Coherence: 0.000
Total Signals: 0.0
```

**Diagnosis:** You need the latest code with the API fix.

**Solution:**
```bash
# Pull the latest code
git pull origin claude/mycoagent-resilience-setup-01JPQq2DCngyxmZey6TVvqsJ

# Verify you have commit aeb8d69 or later
git log --oneline -5

# Run with diagnostic logging
python myconet_contemplative_main.py --config basic --max-steps 100
```

**What to look for in logs:**
```
🔍 AGENT DIAGNOSTIC: type=ContemplativeNeuroAgent, has_wisdom_attr=True, has_update=True
✅ Agent 0: Using NEW contemplative API
📊 Agent 0 @ step 100: action=MEDITATE, age=100, energy=0.943, wisdom=3
```

If you see `⚠️ Using LEGACY simple API`, the agents are not using the full contemplative system.

### Expected Healthy Output

A working simulation should show:
```
Average Age: 500.3  (incrementing)
Average Energy: 0.62  (fluctuating, not frozen at 1.0)
Total Wisdom Generated: 231.0  (increasing)
Network Coherence: 0.453  (non-zero)
Total Signals: 1247.0  (increasing)
Wisdom Archive: 47 insights stored
Rituals Executed: 12
```

### Action Distribution

You should see meditation happening:
```
Action Distribution (10000 total actions):
  MOVE_NORTH          : 1234 ( 12.3%)
  EAT_FOOD           :  987 (  9.9%)
  MEDITATE           :  770 (  7.7%)  ← Should be ~7-10%
  REST               :  654 (  6.5%)
  SHARE_WISDOM       :  423 (  4.2%)
  HELP_OTHER         :  312 (  3.1%)
```

If MEDITATE shows 0 or is missing, agents aren't using their brains properly.

## Development Roadmap

### Phase I: Foundation ✅
- Core contemplative processing
- Wisdom signal system
- Basic agent-environment interaction

### Phase II: Agent Enhancement ✅
- Neural decision-making with PyTorch
- Wisdom emission from agents
- Ethical reasoning integration

### Phase III: Overmind Governance ✅
- WisdomArchive for long-term storage
- Adaptive thresholds and regulation
- Ritual scheduling system
- Attention routing and intervention
- NetworkX visualization

### Phase IV: RL Training (In Progress)
- Train overmind policies using stable-baselines3
- Multi-objective optimization
- Transfer learning across colony sizes
- Emergent coordination strategies

### Phase V: Advanced Features (Planned)
- Human-AI collaboration nodes
- Multi-colony federations
- Adversarial resilience testing
- Long-term wisdom accumulation studies

## Research Applications

### Studying Collective Intelligence
- How does wisdom propagate through agent networks?
- What triggers network-wide contemplative states?
- How do ethical behaviors emerge and spread?
- What are the optimal overmind intervention strategies?

### Human-AI Symbiosis Research
- Network integration of human and AI nodes
- Collaborative wisdom generation
- Mutual benefit optimization
- Contemplative AI alignment

### Beneficial AI Development
- Training AI systems with built-in ethical reasoning
- Network-level safety mechanisms
- Contemplative approach to AI alignment
- Multi-agent coordination for beneficial outcomes

## Citation

If you use this code in your research, please cite:

```bibtex
@software{contemplative_myconet_2025,
  title={Contemplative MycoNet++: A Multi-Agent Simulation Framework for Wisdom-Based Collective Intelligence},
  author={},
  year={2025},
  url={https://github.com/DeepakEz/contem_myco},
  note={Phase III Full Stack Implementation with Overmind Governance}
}
```

## Contributing

Contributions are welcome! Areas of interest:
- New contemplative signal types
- Alternative ethical reasoning frameworks
- Overmind intervention strategies
- Visualization improvements
- Performance optimizations
- Documentation and tutorials

## License

MIT License (see LICENSE file)

---

*"True intelligence is wise intelligence, and true wisdom is efficient wisdom."*

*"The mycelial network teaches us: wisdom flows where connection allows."*
