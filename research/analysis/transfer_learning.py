"""
Transfer Learning Evaluation
==============================
Evaluates whether learned ethical constraints, communication patterns,
and mindfulness behaviors transfer across tasks.

Includes:
- Cross-environment transfer testing
- Module-level transfer analysis
- Fine-tuning efficiency
- Zero-shot transfer metrics
- Transfer learning curves
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import copy
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TransferResult:
    """Result of a transfer learning experiment."""
    source_task: str
    target_task: str
    transfer_type: str
    zero_shot_reward: float
    fine_tuned_reward: float
    from_scratch_reward: float
    transfer_ratio: float  # fine_tuned / from_scratch
    jumpstart: float  # zero_shot - random_init
    time_to_threshold: Optional[float]
    fine_tune_steps: int
    module_contributions: Dict[str, float]


class TransferLearningEvaluator:
    """
    Evaluates transfer of learned modules across environments.

    Tests whether:
    1. Ethical constraints learned in one environment apply in another
    2. Communication protocols transfer between different team compositions
    3. Mindfulness calibration transfers under different noise distributions
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def evaluate_transfer(
        self,
        source_agent_state: Dict,
        source_config,
        target_env,
        target_config,
        fine_tune_steps: int = 50000,
        n_eval_episodes: int = 20,
    ) -> TransferResult:
        """
        Evaluate transfer from source task to target task.

        Args:
            source_agent_state: State dict from trained source agent
            source_config: Config used for source training
            target_env: Target environment
            target_config: Config for target task
            fine_tune_steps: Steps for fine-tuning
            n_eval_episodes: Evaluation episodes
        """
        from research.agents import MultiAgentSystem
        from research.training import MAPPOTrainer

        # 1. Zero-shot evaluation (load source weights, evaluate on target)
        zero_shot_agent = MultiAgentSystem(
            num_agents=target_config.env.num_agents,
            obs_size=target_env.obs_size,
            action_size=target_env.action_size,
            config=source_config,
            share_params=True,
        )

        # Load what we can from source (handle dimension mismatches gracefully)
        self._load_compatible_weights(zero_shot_agent, source_agent_state)
        zero_shot_reward = self._evaluate(zero_shot_agent, target_env, n_eval_episodes)

        # 2. Fine-tune evaluation
        fine_tune_agent = MultiAgentSystem(
            num_agents=target_config.env.num_agents,
            obs_size=target_env.obs_size,
            action_size=target_env.action_size,
            config=target_config,
            share_params=True,
        )
        self._load_compatible_weights(fine_tune_agent, source_agent_state)

        fine_tune_config = copy.deepcopy(target_config)
        fine_tune_config.training.total_timesteps = fine_tune_steps
        fine_tune_config.training.learning_rate = target_config.training.learning_rate * 0.1

        trainer = MAPPOTrainer(
            env=target_env,
            agent_system=fine_tune_agent,
            config=fine_tune_config,
            device=self.device,
        )
        trainer.train(total_timesteps=fine_tune_steps)
        fine_tuned_reward = self._evaluate(fine_tune_agent, target_env, n_eval_episodes)

        # 3. From-scratch evaluation
        scratch_agent = MultiAgentSystem(
            num_agents=target_config.env.num_agents,
            obs_size=target_env.obs_size,
            action_size=target_env.action_size,
            config=target_config,
            share_params=True,
        )
        scratch_config = copy.deepcopy(target_config)
        scratch_config.training.total_timesteps = fine_tune_steps

        scratch_trainer = MAPPOTrainer(
            env=target_env,
            agent_system=scratch_agent,
            config=scratch_config,
            device=self.device,
        )
        scratch_trainer.train(total_timesteps=fine_tune_steps)
        from_scratch_reward = self._evaluate(scratch_agent, target_env, n_eval_episodes)

        # Compute transfer metrics
        transfer_ratio = fine_tuned_reward / (abs(from_scratch_reward) + 1e-12)
        jumpstart = zero_shot_reward

        # Module contributions via ablation transfer
        module_contribs = self._evaluate_module_transfer(
            source_agent_state, source_config, target_env, target_config, n_eval_episodes
        )

        return TransferResult(
            source_task=source_config.name,
            target_task=target_config.name,
            transfer_type="full",
            zero_shot_reward=float(zero_shot_reward),
            fine_tuned_reward=float(fine_tuned_reward),
            from_scratch_reward=float(from_scratch_reward),
            transfer_ratio=float(transfer_ratio),
            jumpstart=float(jumpstart),
            time_to_threshold=None,
            fine_tune_steps=fine_tune_steps,
            module_contributions=module_contribs,
        )

    def _evaluate_module_transfer(
        self,
        source_state: Dict,
        source_config,
        target_env,
        target_config,
        n_episodes: int,
    ) -> Dict[str, float]:
        """
        Evaluate which modules contribute most to transfer.

        Transfers one module at a time and measures benefit.
        """
        from research.agents import MultiAgentSystem

        contributions = {}
        modules = ["ethics", "diffusion", "mindfulness"]

        # Baseline: no transfer
        baseline_agent = MultiAgentSystem(
            num_agents=target_config.env.num_agents,
            obs_size=target_env.obs_size,
            action_size=target_env.action_size,
            config=target_config,
            share_params=True,
        )
        baseline_reward = self._evaluate(baseline_agent, target_env, n_episodes)

        for module_name in modules:
            try:
                # Create agent and transfer only this module's weights
                agent = MultiAgentSystem(
                    num_agents=target_config.env.num_agents,
                    obs_size=target_env.obs_size,
                    action_size=target_env.action_size,
                    config=target_config,
                    share_params=True,
                )
                self._load_module_weights(agent, source_state, module_name)
                module_reward = self._evaluate(agent, target_env, n_episodes)
                contributions[module_name] = float(module_reward - baseline_reward)
            except Exception as e:
                logger.warning(f"Could not evaluate transfer for {module_name}: {e}")
                contributions[module_name] = 0.0

        return contributions

    def _load_compatible_weights(self, agent_system, state_dict: Dict):
        """Load weights that are dimensionally compatible."""
        target_state = agent_system.state_dict()
        loaded = 0
        skipped = 0

        for key, param in state_dict.items():
            if key in target_state:
                if isinstance(param, torch.Tensor) and isinstance(target_state[key], torch.Tensor):
                    if param.shape == target_state[key].shape:
                        target_state[key] = param
                        loaded += 1
                    else:
                        skipped += 1

        agent_system.load_state_dict(target_state)
        logger.info(f"Loaded {loaded} parameters, skipped {skipped} (shape mismatch)")

    def _load_module_weights(self, agent_system, state_dict: Dict, module_name: str):
        """Load only weights for a specific module."""
        target_state = agent_system.state_dict()

        for key, param in state_dict.items():
            if module_name in key:
                if key in target_state and isinstance(param, torch.Tensor):
                    if param.shape == target_state[key].shape:
                        target_state[key] = param

        agent_system.load_state_dict(target_state)

    def _evaluate(self, agent_system, env, n_episodes: int) -> float:
        """Run evaluation episodes and return mean reward."""
        total_rewards = []

        for _ in range(n_episodes):
            obs_dict, _ = env.reset()
            agent_system.reset()
            ep_reward = 0.0
            done = False

            while not done:
                outputs = agent_system.step(obs_dict, deterministic=True)
                actions = {aid: int(out.action.item()) for aid, out in outputs.items()}
                obs_dict, rewards, terms, truncs, _ = env.step(actions)
                ep_reward += sum(rewards.values())
                done = any(terms.values()) or any(truncs.values())

            total_rewards.append(ep_reward)

        return float(np.mean(total_rewards))

    def plot_transfer_matrix(
        self,
        transfer_results: List[TransferResult],
        save_path: Optional[str] = None,
    ):
        """Plot transfer learning result matrix."""
        if not MATPLOTLIB_AVAILABLE:
            return

        sources = sorted(set(r.source_task for r in transfer_results))
        targets = sorted(set(r.target_task for r in transfer_results))

        matrix = np.zeros((len(sources), len(targets)))
        for r in transfer_results:
            i = sources.index(r.source_task)
            j = targets.index(r.target_task)
            matrix[i, j] = r.transfer_ratio

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel("Target Task")
        ax.set_ylabel("Source Task")

        for i in range(len(sources)):
            for j in range(len(targets)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                       ha="center", va="center", fontsize=10,
                       color="black" if 0.3 < matrix[i, j] < 1.7 else "white")

        plt.colorbar(im, label="Transfer Ratio (fine-tuned / from-scratch)")
        plt.title("Cross-Task Transfer Matrix", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
