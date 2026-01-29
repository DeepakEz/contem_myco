"""
Test script to verify all modules work correctly.
Run: python -m research.test_modules
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add research to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.config import ExperimentConfig, get_full_config, get_ablation_configs
from research.agents.modules import (
    DiffusionField,
    DiffusionEncoder,
    create_ethics_module,
    create_mindfulness_module,
)
from research.agents import ContemplativeAgent, MultiAgentSystem


def test_config():
    """Test configuration system."""
    print("Testing configuration system...")

    config = get_full_config()
    assert config.diffusion.enabled
    assert config.ethics.enabled
    assert config.mindfulness.enabled

    ablations = get_ablation_configs()
    assert len(ablations) == 5  # full, no_ethics, no_diffusion, no_mindfulness, baseline

    print("  [OK] Configuration system works")


def test_diffusion_field():
    """Test diffusion field module."""
    print("Testing diffusion field...")

    field = DiffusionField(grid_size=32, num_channels=4)

    # Test deposit
    field.deposit(np.array([0.0, 0.0]), channel=0, strength=1.0)
    assert field.field[0].max() > 0

    # Test sense
    obs = field.sense(np.array([0.0, 0.0]))
    assert obs.shape == (12,)  # 3 features * 4 channels
    assert obs[0] > 0  # Should sense deposited signal

    # Test step (diffusion + decay)
    initial_max = field.field[0].max()
    field.step()
    assert field.field[0].max() < initial_max  # Should decay

    print("  [OK] Diffusion field works")


def test_diffusion_encoder():
    """Test diffusion encoder network."""
    print("Testing diffusion encoder...")

    encoder = DiffusionEncoder(num_channels=4)

    field_obs = torch.randn(8, 12)  # batch of 8, 12 features
    encoded = encoder(field_obs)

    assert encoded.shape == (8, 32)  # Default output size

    print("  [OK] Diffusion encoder works")


def test_ethics_module():
    """Test ethics module."""
    print("Testing ethics module...")

    config = get_full_config()
    ethics = create_ethics_module(config.ethics, state_size=16, action_size=5)

    state = torch.randn(8, 16)
    action = torch.randn(8, 5)

    score, framework_scores = ethics(state, action)

    assert score.shape == (8, 1)
    assert len(framework_scores) == 4

    # Test constraint checking
    violations = ethics.constraints.evaluate_all(
        action_effects={"harm_caused": 0.5},
        rewards=np.array([1.0, 2.0, 3.0, 10.0]),  # Unequal
        cooperation_rate=0.1
    )
    assert violations["harm"][0]  # Should be violated
    assert violations["fairness"][0]  # Should be violated

    print("  [OK] Ethics module works")


def test_mindfulness_module():
    """Test mindfulness module."""
    print("Testing mindfulness module...")

    config = get_full_config()
    mindfulness = create_mindfulness_module(
        config.mindfulness,
        obs_size=16,
        action_size=5
    )

    obs = torch.randn(8, 16)
    action = torch.randn(8, 5)
    next_obs = torch.randn(8, 16)

    # Test prediction
    pred, uncertainty = mindfulness.ensemble(obs, action)
    assert pred.shape == (8, 16)
    assert uncertainty.shape == (8,)

    # Test surprise
    surprise = mindfulness.ensemble.compute_surprise(obs, action, next_obs)
    assert surprise.shape == (8,)

    # Test mindfulness state
    state = mindfulness.compute_mindfulness_state(obs, action, next_obs)
    assert hasattr(state, 'surprise_level')
    assert hasattr(state, 'uncertainty')

    print("  [OK] Mindfulness module works")


def test_contemplative_agent():
    """Test full contemplative agent."""
    print("Testing contemplative agent...")

    config = get_full_config()

    agent = ContemplativeAgent(
        obs_size=16,
        action_size=5,
        config=config,
        continuous=False,
    )

    obs = torch.randn(1, 16)

    # Test without diffusion field
    output = agent(obs, deterministic=True)
    assert output.action is not None
    assert output.value is not None

    # Test with diffusion field
    field = DiffusionField(
        grid_size=config.diffusion.grid_size,
        num_channels=config.diffusion.num_channels
    )
    position = np.array([0.0, 0.0])

    output = agent(obs, diffusion_field=field, position=position)
    assert output.action is not None
    assert output.deposit_strengths is not None

    print("  [OK] Contemplative agent works")


def test_multi_agent_system():
    """Test multi-agent system."""
    print("Testing multi-agent system...")

    config = get_full_config()
    config.env.num_agents = 4

    system = MultiAgentSystem(
        num_agents=4,
        obs_size=16,
        action_size=5,
        config=config,
        share_params=True,
    )

    observations = {
        f"agent_{i}": np.random.randn(16).astype(np.float32)
        for i in range(4)
    }
    positions = {
        f"agent_{i}": np.random.uniform(-1, 1, 2).astype(np.float32)
        for i in range(4)
    }

    # Test step
    outputs = system.step(observations, positions)
    assert len(outputs) == 4

    for agent_id, output in outputs.items():
        assert output.action is not None
        assert output.value is not None

    # Test diffusion field updated
    assert system.diffusion_field.field.sum() > 0  # Some deposits made

    print("  [OK] Multi-agent system works")


def test_ablations():
    """Test that ablation configs work."""
    print("Testing ablation configurations...")

    configs = get_ablation_configs()

    for name, config in configs.items():
        agent = ContemplativeAgent(
            obs_size=16,
            action_size=5,
            config=config,
            continuous=False,
        )

        obs = torch.randn(1, 16)
        output = agent(obs, deterministic=True)
        assert output.action is not None

        print(f"  [OK] {name} configuration works")

    print("  [OK] All ablations work")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONTEMPLATIVE MARL - MODULE TESTS")
    print("=" * 60)
    print()

    test_config()
    test_diffusion_field()
    test_diffusion_encoder()
    test_ethics_module()
    test_mindfulness_module()
    test_contemplative_agent()
    test_multi_agent_system()
    test_ablations()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
