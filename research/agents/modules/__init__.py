"""
Contemplative MARL Modules
==========================
Core modules for the Contemplative MARL framework:

- Module A (Ethics): Multi-framework ethical constraints with Lagrangian optimization
- Module B (Diffusion): Stigmergic communication via spatial signal fields
- Module C (Mindfulness): Uncertainty-aware gating for robustness
- Dynamic Ethics: Context-dependent ethical framework weighting
- Interpretable Ethics: Attention-based explanations for ethical decisions
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

from .dynamic_ethics import (
    DynamicMultiFrameworkEthics,
    ContextualWeightNetwork,
)

from .interpretable_ethics import (
    AttentiveEthicsEvaluator,
    EthicsAuditTrail,
    EthicalExplanation,
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
    # Dynamic Ethics
    "DynamicMultiFrameworkEthics",
    "ContextualWeightNetwork",
    # Interpretable Ethics
    "AttentiveEthicsEvaluator",
    "EthicsAuditTrail",
    "EthicalExplanation",
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
