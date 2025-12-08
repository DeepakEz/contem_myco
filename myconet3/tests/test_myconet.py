"""
MycoNet 3.0 Test Suite
======================

Unit tests and integration tests for all MycoNet 3.0 components.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myconet3.config import (
    MycoNetConfig,
    create_minimal_config,
    create_basic_config,
    WisdomSignalType
)
from myconet3.environment import Environment, WisdomSignalGrid
from myconet3.uprt_field import UPRTField
from myconet3.field_metrics import MetricsComputer, ColonyMetrics
from myconet3.hypernetwork import GenomeHyperNet, EvolutionEngine, TargetArchitecture
from myconet3.symbolic_parametric_bridge import SymbolicParametricBridge, PatchGate
from myconet3.myco_agent import MycoAgent
from myconet3.overmind import Overmind, DharmaCompiler
from myconet3.training_pipeline import TrainingPipeline, RolloutBuffer
from myconet3.scenarios import ScenarioRunner, ScenarioType
from myconet3.evaluation import Evaluator, TheoreticalPredictionValidator


class TestConfiguration(unittest.TestCase):
    """Tests for configuration module."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = MycoNetConfig()
        self.assertEqual(config.num_agents, 20)
        self.assertEqual(config.environment.grid_size, (64, 64))

    def test_minimal_config(self):
        """Test minimal configuration preset."""
        config = create_minimal_config()
        self.assertEqual(config.num_agents, 5)
        self.assertEqual(config.environment.grid_size, (32, 32))

    def test_config_validation(self):
        """Test configuration validation."""
        config = MycoNetConfig()
        self.assertTrue(config.validate())

        # Invalid config
        config.environment.grid_size = (-1, 64)
        self.assertFalse(config.validate())

    def test_config_serialization(self):
        """Test config to/from dict."""
        config = MycoNetConfig()
        config.num_agents = 30
        config_dict = config.to_dict()

        restored = MycoNetConfig.from_dict(config_dict)
        self.assertEqual(restored.num_agents, 30)


class TestEnvironment(unittest.TestCase):
    """Tests for environment module."""

    def setUp(self):
        self.config = create_minimal_config()
        self.env = Environment(self.config)

    def test_environment_creation(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.grid_size, (32, 32))

    def test_position_validation(self):
        """Test position validation."""
        self.assertTrue(self.env.is_valid_position((10, 10)))
        self.assertFalse(self.env.is_valid_position((-1, 10)))
        self.assertFalse(self.env.is_valid_position((100, 10)))

    def test_signal_emission(self):
        """Test wisdom signal emission."""
        self.env.emit_signal((16, 16), signal_type=0, intensity=1.0)
        signals = self.env.get_local_signals((16, 16), radius=2)
        self.assertGreater(signals.sum(), 0)

    def test_signal_propagation(self):
        """Test signal diffusion."""
        self.env.emit_signal((16, 16), signal_type=0, intensity=1.0)
        initial_value = self.env.get_local_signals((16, 16), radius=0).sum()

        for _ in range(10):
            self.env.propagate_signals()

        # Signal should have diffused
        final_value = self.env.get_local_signals((16, 16), radius=0).sum()
        self.assertLess(final_value, initial_value)

    def test_resource_management(self):
        """Test resource addition and harvesting."""
        self.env.add_resource((10, 10), amount=1.0)
        resource = self.env.get_resource_at((10, 10))
        self.assertGreater(resource, 0)

        harvested = self.env.harvest_resource((10, 10))
        self.assertGreater(harvested, 0)


class TestUPRTField(unittest.TestCase):
    """Tests for UPRT field module."""

    def setUp(self):
        self.config = create_minimal_config()
        self.field = UPRTField(self.config)

    def test_field_initialization(self):
        """Test field initialization."""
        self.assertIsNotNone(self.field)
        state = self.field.get_field_state()
        self.assertEqual(state.shape[0], self.config.uprt_field.field_resolution)

    def test_field_step(self):
        """Test field dynamics step."""
        agent_states = np.random.randn(5, 10)
        initial_state = self.field.get_field_state().copy()

        self.field.step(agent_states)

        final_state = self.field.get_field_state()
        self.assertFalse(np.allclose(initial_state, final_state))

    def test_topological_defect_detection(self):
        """Test defect detection."""
        defects = self.field.detect_defects()
        self.assertIsInstance(defects, list)


class TestFieldMetrics(unittest.TestCase):
    """Tests for field metrics computation."""

    def setUp(self):
        self.config = create_minimal_config()
        self.computer = MetricsComputer(self.config)

    def test_metrics_computation(self):
        """Test computing all metrics."""
        state_history = np.random.randn(20, 5, 10)
        field_state = np.random.randn(64, 64)

        metrics = self.computer.compute_all(state_history, field_state)

        self.assertIsInstance(metrics, ColonyMetrics)
        self.assertGreaterEqual(metrics.entropy, 0)
        self.assertGreaterEqual(metrics.coherence, 0)
        self.assertLessEqual(metrics.coherence, 1)

    def test_entropy_bounds(self):
        """Test entropy is properly bounded."""
        state_history = np.random.randn(20, 5, 10)
        field_state = np.random.randn(64, 64)

        metrics = self.computer.compute_all(state_history, field_state)
        self.assertGreaterEqual(metrics.entropy, 0)

    def test_coherence_bounds(self):
        """Test coherence (Kuramoto) is in [0, 1]."""
        state_history = np.random.randn(20, 5, 10)
        field_state = np.random.randn(64, 64)

        metrics = self.computer.compute_all(state_history, field_state)
        self.assertGreaterEqual(metrics.coherence, 0)
        self.assertLessEqual(metrics.coherence, 1)


class TestHypernetwork(unittest.TestCase):
    """Tests for hypernetwork and evolution."""

    def setUp(self):
        self.config = create_minimal_config()
        self.target_arch = TargetArchitecture(
            input_dim=100,
            hidden_dim=64,
            output_dim=15
        )

    def test_evolution_engine(self):
        """Test evolution engine."""
        engine = EvolutionEngine(self.config)

        population = engine.initialize_population()
        self.assertEqual(len(population), self.config.qrea.population_size)

        fitness = np.random.randn(len(population))
        new_pop = engine.evolve(population, fitness)
        self.assertEqual(len(new_pop), len(population))


class TestSymbolicBridge(unittest.TestCase):
    """Tests for symbolic-parametric bridge."""

    def setUp(self):
        self.config = create_minimal_config()
        self.bridge = SymbolicParametricBridge(self.config)

    def test_bridge_creation(self):
        """Test bridge initialization."""
        self.assertIsNotNone(self.bridge)

    def test_patch_gate_safety(self):
        """Test PatchGate safety validation."""
        gate = PatchGate(self.config)

        # Test safe patch
        safe_patch = {'magnitude': 0.01, 'reversible': True}
        result = gate.validate(safe_patch)
        self.assertTrue(result.safe)

        # Test unsafe patch
        unsafe_patch = {'magnitude': 100.0, 'reversible': False}
        result = gate.validate(unsafe_patch)
        self.assertFalse(result.safe)


class TestMycoAgent(unittest.TestCase):
    """Tests for MycoAgent."""

    def setUp(self):
        self.config = create_minimal_config()
        self.agent = MycoAgent(self.config, agent_id=0)

    def test_agent_creation(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.agent_id, 0)
        self.assertTrue(self.agent.alive)

    def test_agent_reset(self):
        """Test agent reset."""
        self.agent.energy = 0.1
        self.agent.reset(position=(10, 10))

        self.assertEqual(self.agent.position, (10, 10))
        self.assertEqual(self.agent.energy, self.config.agent.initial_energy)

    def test_state_vector(self):
        """Test state vector generation."""
        state = self.agent.get_state_vector()
        self.assertIsInstance(state, np.ndarray)
        self.assertGreater(len(state), 0)

    def test_action_selection(self):
        """Test action selection."""
        obs = np.random.randn(100).astype(np.float32)
        action, value, log_prob = self.agent.select_action(obs)

        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)


class TestOvermind(unittest.TestCase):
    """Tests for Overmind coordinator."""

    def setUp(self):
        self.config = create_minimal_config()
        self.overmind = Overmind(self.config)

    def test_overmind_creation(self):
        """Test overmind initialization."""
        self.assertIsNotNone(self.overmind)

    def test_overmind_step(self):
        """Test overmind step."""
        agent_states = [np.random.randn(10) for _ in range(5)]
        field_state = np.random.randn(64, 64)

        self.overmind.step(agent_states, field_state)

    def test_dharma_compiler(self):
        """Test Dharma Compiler."""
        compiler = DharmaCompiler(self.config)
        self.assertIsNotNone(compiler)


class TestTrainingPipeline(unittest.TestCase):
    """Tests for training pipeline."""

    def setUp(self):
        self.config = create_minimal_config()
        self.config.training.rl_episodes = 2
        self.config.training.rl_max_steps = 10
        self.config.training.evolution_generations = 2

    def test_rollout_buffer(self):
        """Test rollout buffer."""
        buffer = RolloutBuffer()

        # Add some transitions
        for _ in range(10):
            buffer.add(
                state=np.random.randn(100),
                action=np.random.randint(0, 15),
                reward=np.random.randn(),
                value=np.random.randn(),
                log_prob=np.random.randn(),
                done=False
            )

        self.assertEqual(len(buffer.states), 10)

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=0.99, lam=0.95
        )
        self.assertEqual(len(returns), 10)


class TestScenarios(unittest.TestCase):
    """Tests for test scenarios."""

    def setUp(self):
        self.config = create_minimal_config()
        self.runner = ScenarioRunner(self.config)

    def test_scenario_creation(self):
        """Test scenario creation."""
        for scenario_type in ScenarioType:
            scenario = self.runner.create_scenario(scenario_type)
            self.assertIsNotNone(scenario)

    def test_resource_foraging_setup(self):
        """Test resource foraging scenario setup."""
        from myconet3.scenarios import ResourceForagingScenario

        scenario = ResourceForagingScenario(self.config)
        self.assertEqual(scenario.scenario_type, ScenarioType.RESOURCE_FORAGING)


class TestEvaluation(unittest.TestCase):
    """Tests for evaluation framework."""

    def setUp(self):
        self.config = create_minimal_config()
        self.evaluator = Evaluator(self.config)

    def test_evaluator_creation(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)

    def test_prediction_validator(self):
        """Test theoretical prediction validator."""
        validator = TheoreticalPredictionValidator(self.config)

        metrics = {
            'entropy': 0.5,
            'mutual_information': 0.7,
            'coherence': 0.6,
            'integrated_information': 0.3,
            'phenomenal_curvature': 0.2,
            'recoverability_index': 0.8,
            'resonance_half_life': 15.0
        }

        validations = validator.validate_all(metrics)
        self.assertGreater(len(validations), 0)

    def test_colony_evaluation(self):
        """Test colony evaluation."""
        state_history = np.random.randn(20, 5, 10)
        field_states = np.random.randn(20, 64, 64)

        result = self.evaluator.evaluate_colony(state_history, field_states)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for full system."""

    def setUp(self):
        self.config = create_minimal_config()
        self.config.training.rl_episodes = 1
        self.config.training.rl_max_steps = 5

    def test_full_simulation_step(self):
        """Test running a full simulation step."""
        from myconet3.main import MycoNetSimulation

        sim = MycoNetSimulation(self.config)
        result = sim.step()

        self.assertIn('step', result)
        self.assertIn('alive_agents', result)
        self.assertGreater(result['alive_agents'], 0)

    def test_episode_run(self):
        """Test running a complete episode."""
        from myconet3.main import MycoNetSimulation

        sim = MycoNetSimulation(self.config)
        result = sim.run_episode(max_steps=10)

        self.assertIn('total_reward', result)
        self.assertIn('steps', result)

    def test_component_integration(self):
        """Test that all components work together."""
        # Create all components
        env = Environment(self.config)
        field = UPRTField(self.config)
        overmind = Overmind(self.config)
        metrics = MetricsComputer(self.config)

        agents = [MycoAgent(self.config, agent_id=i) for i in range(3)]

        # Initialize agents
        for agent in agents:
            pos = env.get_random_valid_position()
            agent.reset(position=pos)

        # Run several steps
        for _ in range(5):
            # Agents act
            for agent in agents:
                obs = np.random.randn(100).astype(np.float32)
                action, _, _ = agent.select_action(obs)

            # Update environment
            env.propagate_signals()

            # Update field
            agent_states = np.array([a.get_state_vector() for a in agents])
            field.step(agent_states)

            # Overmind reflection
            field_state = field.get_field_state()
            overmind.step([a.get_state_vector() for a in agents], field_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
