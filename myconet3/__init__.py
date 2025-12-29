"""
MycoNet 3.0: A Flexible Cognitive Architecture for Distributed Agents
=====================================================================

Bio-inspired multi-agent AI system with:
- Swarm intelligence and distributed cognition
- Field-theoretic cognitive modeling (UPRT)
- Contemplative AI principles with ethical constraints
- Multi-scale optimization (RL + Evolution + Surrogates)

Main Components:
- MycoAgent: Individual cognitive agents with mindfulness monitoring
- Overmind: Central coordinator with UPRT field integration
- Environment: 2D grid with resources and wisdom signal propagation
- Training Pipeline: Nested optimization loops
- Evaluation: Comprehensive validation framework

Usage:
    from myconet3 import MycoNetSimulation, MycoNetConfig

    # Quick demo
    sim = MycoNetSimulation()
    sim.run_demo(num_steps=100)

    # Full training
    config = MycoNetConfig()
    config.num_agents = 50
    sim = MycoNetSimulation(config)
    results = sim.train()

    # Evaluation
    result = sim.evaluate()
    print(f"Score: {result.overall_score}")
"""

__version__ = "3.0.0"
__author__ = "MycoNet Team"

# Configuration
from .config import (
    MycoNetConfig,
    EnvironmentConfig,
    AgentConfig,
    UPRTFieldConfig,
    OvermindConfig,
    QREAConfig,
    DharmaConfig,
    TrainingConfig,
    MetricsConfig,
    WisdomSignalType,
    InsightType,
    create_minimal_config,
    create_basic_config,
    create_advanced_config,
    create_scalability_test_config
)

# Environment
from .environment import (
    Environment,
    WisdomSignalGrid,
    ResourcePatch,
    TerrainType,
    StochasticEvent
)

# UPRT Field
from .uprt_field import (
    UPRTField,
    TopologicalDefect,
    FieldState
)

# Field Metrics
from .field_metrics import (
    ColonyMetrics,
    MetricsComputer
)

# Hypernetwork and Evolution
from .hypernetwork import (
    GenomeHyperNet,
    EvolutionEngine,
    HierarchicalMERA,
    TargetArchitecture,
    LayerSpec
)

# Symbolic-Parametric Bridge
from .symbolic_parametric_bridge import (
    SymbolicParametricBridge,
    SymbolicReasoner,
    ParameterPatcher,
    PatchGate,
    PatchRisk,
    PatchType,
    ParameterPatch,
    PatchValidationResult
)

# Agents
from .myco_agent import (
    MycoAgent,
    WisdomInsight,
    AgentState,
    ActionType
)

# Overmind
from .overmind import (
    Overmind,
    DharmaCompiler,
    InterventionType,
    DetectedSymbol
)

# Training
from .training_pipeline import (
    TrainingPipeline,
    TrainingMetrics,
    RolloutBuffer,
    PPOTrainer,
    SurrogateCalibrator,
    FitnessEvaluator,
    create_training_pipeline
)

# Scenarios
from .scenarios import (
    ScenarioType,
    ScenarioResult,
    ScenarioRunner,
    ResourceForagingScenario,
    TrolleyDilemmaScenario,
    DisasterRecoveryScenario,
    NovelConceptEmergenceScenario,
    ScalabilityStressScenario,
    create_scenario
)

# Evaluation
from .evaluation import (
    Evaluator,
    EvaluationResult,
    PredictionValidation,
    TheoreticalPredictionValidator,
    EthicalComplianceChecker,
    ScalabilityAssessor,
    create_evaluator,
    quick_evaluate
)

# Main orchestrator
from .main import (
    MycoNetSimulation,
    run_simulation,
    main
)

__all__ = [
    # Version
    '__version__',

    # Configuration
    'MycoNetConfig',
    'EnvironmentConfig',
    'AgentConfig',
    'UPRTFieldConfig',
    'OvermindConfig',
    'QREAConfig',
    'DharmaConfig',
    'TrainingConfig',
    'MetricsConfig',
    'WisdomSignalType',
    'InsightType',
    'create_minimal_config',
    'create_basic_config',
    'create_advanced_config',
    'create_scalability_test_config',

    # Environment
    'Environment',
    'WisdomSignalGrid',
    'ResourcePatch',
    'TerrainType',
    'StochasticEvent',

    # UPRT Field
    'UPRTField',
    'TopologicalDefect',
    'FieldState',

    # Metrics
    'ColonyMetrics',
    'MetricsComputer',

    # Hypernetwork
    'GenomeHyperNet',
    'EvolutionEngine',
    'HierarchicalMERA',
    'TargetArchitecture',
    'LayerSpec',

    # Symbolic Bridge
    'SymbolicParametricBridge',
    'SymbolicReasoner',
    'ParameterPatcher',
    'PatchGate',
    'PatchRisk',
    'PatchType',
    'ParameterPatch',
    'PatchValidationResult',

    # Agents
    'MycoAgent',
    'WisdomInsight',
    'AgentState',
    'ActionType',

    # Overmind
    'Overmind',
    'DharmaCompiler',
    'InterventionType',
    'DetectedSymbol',

    # Training
    'TrainingPipeline',
    'TrainingMetrics',
    'RolloutBuffer',
    'PPOTrainer',
    'SurrogateCalibrator',
    'FitnessEvaluator',
    'create_training_pipeline',

    # Scenarios
    'ScenarioType',
    'ScenarioResult',
    'ScenarioRunner',
    'ResourceForagingScenario',
    'TrolleyDilemmaScenario',
    'DisasterRecoveryScenario',
    'NovelConceptEmergenceScenario',
    'ScalabilityStressScenario',
    'create_scenario',

    # Evaluation
    'Evaluator',
    'EvaluationResult',
    'PredictionValidation',
    'TheoreticalPredictionValidator',
    'EthicalComplianceChecker',
    'ScalabilityAssessor',
    'create_evaluator',
    'quick_evaluate',

    # Main
    'MycoNetSimulation',
    'run_simulation',
    'main'
]
