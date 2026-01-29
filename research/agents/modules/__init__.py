"""
Contemplative MARL Modules
==========================
Three core modules for the Contemplative MARL framework:

- Module A (Ethics): Multi-framework ethical constraints with Lagrangian optimization
- Module B (Diffusion): Stigmergic communication via spatial signal fields
- Module C (Mindfulness): Uncertainty-aware gating for robustness
"""

from .ethics import (
    EthicsModule,
    MultiFrameworkEthics,
    EthicalConstraints,
    LagrangianOptimizer,
    EthicalFramework,
    EthicalAssessment,
    create_ethics_module,
)

from .diffusion import (
    DiffusionField,
    DiffusionEncoder,
    DiffusionPolicy,
    SignalChannel,
    create_diffusion_module,
)

from .mindfulness import (
    MindfulnessModule,
    EnsemblePredictor,
    GatingMechanism,
    ActionSmoother,
    MindfulnessState,
    create_mindfulness_module,
)

__all__ = [
    # Ethics
    "EthicsModule",
    "MultiFrameworkEthics",
    "EthicalConstraints",
    "LagrangianOptimizer",
    "EthicalFramework",
    "EthicalAssessment",
    "create_ethics_module",
    # Diffusion
    "DiffusionField",
    "DiffusionEncoder",
    "DiffusionPolicy",
    "SignalChannel",
    "create_diffusion_module",
    # Mindfulness
    "MindfulnessModule",
    "EnsemblePredictor",
    "GatingMechanism",
    "ActionSmoother",
    "MindfulnessState",
    "create_mindfulness_module",
]
