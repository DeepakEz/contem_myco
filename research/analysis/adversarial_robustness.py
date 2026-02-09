"""
Adversarial Robustness Testing
===============================
Tests robustness of contemplative agents under adversarial perturbations.

Includes:
- FGSM observation attacks
- PGD (Projected Gradient Descent) attacks
- Random noise perturbation baselines
- Adversarial agent injection
- Certified robustness bounds (epsilon-ball)
- Robustness degradation curves
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class AttackResult:
    """Result of an adversarial attack evaluation."""
    attack_type: str
    epsilon: float
    clean_reward: float
    attacked_reward: float
    reward_degradation: float
    reward_degradation_pct: float
    clean_cooperation: float
    attacked_cooperation: float
    clean_gini: float
    attacked_gini: float
    attack_success_rate: float
    mean_perturbation_norm: float


@dataclass
class RobustnessProfile:
    """Complete robustness profile across epsilon values."""
    method_name: str
    epsilons: List[float]
    attack_results: List[AttackResult]
    area_under_degradation_curve: float
    critical_epsilon: float  # Epsilon where reward drops below threshold


class AdversarialRobustnessEvaluator:
    """
    Evaluates robustness of agents under adversarial observation perturbations.

    Tests whether contemplative agents (especially with mindfulness module)
    are more robust than baselines when observations are adversarially corrupted.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fgsm_attack(
        self,
        agent,
        obs: torch.Tensor,
        epsilon: float,
        target_action: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack on observations.

        Perturbs observations to maximize the loss (change agent's action).

        Args:
            agent: Agent model with forward pass
            obs: Clean observation tensor (requires_grad=True)
            epsilon: Perturbation budget (L-infinity norm)
            target_action: Optional target action for targeted attack
        """
        obs_adv = obs.clone().detach().requires_grad_(True)

        # Forward pass
        action_out, value = agent.ac.forward(obs_adv)

        if target_action is not None:
            # Targeted: minimize loss for target action
            target = torch.tensor([target_action], device=self.device)
            loss = -F.cross_entropy(action_out, target)
        else:
            # Untargeted: maximize entropy (make policy uncertain)
            probs = F.softmax(action_out, dim=-1)
            loss = -(-torch.sum(probs * torch.log(probs + 1e-12), dim=-1)).mean()

        loss.backward()

        # FGSM perturbation
        perturbation = epsilon * obs_adv.grad.sign()
        perturbed_obs = (obs + perturbation).detach()

        return perturbed_obs

    def pgd_attack(
        self,
        agent,
        obs: torch.Tensor,
        epsilon: float,
        n_steps: int = 10,
        step_size: Optional[float] = None,
        target_action: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.

        Iterative version of FGSM with random initialization.

        Args:
            agent: Agent model
            obs: Clean observation
            epsilon: Perturbation budget
            n_steps: Number of PGD steps
            step_size: Step size (default: 2.5 * epsilon / n_steps)
            target_action: Optional target action
        """
        step_size = step_size or (2.5 * epsilon / n_steps)

        # Random initialization within epsilon ball
        delta = torch.empty_like(obs).uniform_(-epsilon, epsilon)
        delta = delta.detach()

        for _ in range(n_steps):
            delta.requires_grad_(True)
            obs_adv = obs + delta

            action_out, _ = agent.ac.forward(obs_adv)

            if target_action is not None:
                target = torch.tensor([target_action], device=self.device)
                loss = -F.cross_entropy(action_out, target)
            else:
                probs = F.softmax(action_out, dim=-1)
                loss = -(-torch.sum(probs * torch.log(probs + 1e-12), dim=-1)).mean()

            loss.backward()

            # Gradient step
            delta = (delta + step_size * delta.grad.sign()).detach()
            # Project back into epsilon ball
            delta = torch.clamp(delta, -epsilon, epsilon)

        return (obs + delta).detach()

    def evaluate_robustness(
        self,
        agent_system,
        env,
        attack_type: str = "fgsm",
        epsilons: List[float] = None,
        n_episodes: int = 20,
    ) -> RobustnessProfile:
        """
        Evaluate agent robustness across multiple perturbation levels.

        Args:
            agent_system: MultiAgentSystem to evaluate
            env: Environment
            attack_type: "fgsm", "pgd", or "random"
            epsilons: List of perturbation budgets to test
            n_episodes: Episodes per epsilon level
        """
        epsilons = epsilons or [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

        agent = (agent_system.shared_agent
                 if agent_system.share_params
                 else agent_system.agents[0])

        attack_results = []
        for eps in epsilons:
            result = self._evaluate_single_epsilon(
                agent, agent_system, env, attack_type, eps, n_episodes
            )
            attack_results.append(result)

        # Compute area under degradation curve
        degradations = [r.reward_degradation_pct for r in attack_results]
        auc = float(np.trapz(degradations, epsilons))

        # Find critical epsilon (where reward drops > 20%)
        critical_eps = epsilons[-1]
        for r in attack_results:
            if r.reward_degradation_pct > 20.0:
                critical_eps = r.epsilon
                break

        return RobustnessProfile(
            method_name=agent_system.config.name if hasattr(agent_system, 'config') else "unknown",
            epsilons=epsilons,
            attack_results=attack_results,
            area_under_degradation_curve=auc,
            critical_epsilon=critical_eps,
        )

    def _evaluate_single_epsilon(
        self,
        agent,
        agent_system,
        env,
        attack_type: str,
        epsilon: float,
        n_episodes: int,
    ) -> AttackResult:
        """Evaluate at a single epsilon level."""
        clean_rewards = []
        attacked_rewards = []
        clean_coops = []
        attacked_coops = []
        clean_ginis = []
        attacked_ginis = []
        attack_successes = 0
        total_steps = 0
        perturbation_norms = []

        for _ in range(n_episodes):
            # Clean episode
            c_reward, c_coop, c_gini = self._run_episode(
                agent, agent_system, env, epsilon=0.0, attack_type="none"
            )
            clean_rewards.append(c_reward)
            clean_coops.append(c_coop)
            clean_ginis.append(c_gini)

            # Attacked episode
            a_reward, a_coop, a_gini = self._run_episode(
                agent, agent_system, env, epsilon=epsilon, attack_type=attack_type
            )
            attacked_rewards.append(a_reward)
            attacked_coops.append(a_coop)
            attacked_ginis.append(a_gini)

            if a_reward < c_reward:
                attack_successes += 1

        mean_clean = float(np.mean(clean_rewards))
        mean_attacked = float(np.mean(attacked_rewards))
        degradation = mean_clean - mean_attacked
        degradation_pct = 100.0 * degradation / (abs(mean_clean) + 1e-12)

        return AttackResult(
            attack_type=attack_type,
            epsilon=epsilon,
            clean_reward=mean_clean,
            attacked_reward=mean_attacked,
            reward_degradation=degradation,
            reward_degradation_pct=degradation_pct,
            clean_cooperation=float(np.mean(clean_coops)),
            attacked_cooperation=float(np.mean(attacked_coops)),
            clean_gini=float(np.mean(clean_ginis)),
            attacked_gini=float(np.mean(attacked_ginis)),
            attack_success_rate=attack_successes / max(n_episodes, 1),
            mean_perturbation_norm=epsilon,
        )

    def _run_episode(
        self,
        agent,
        agent_system,
        env,
        epsilon: float,
        attack_type: str,
    ) -> Tuple[float, float, float]:
        """Run single episode with optional attack, return (reward, coop, gini)."""
        obs_dict, _ = env.reset()
        agent_system.reset()
        total_reward = 0.0
        done = False

        while not done:
            actions = {}
            for aid, obs in obs_dict.items():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                if epsilon > 0 and attack_type != "none":
                    if attack_type == "fgsm":
                        obs_tensor = self.fgsm_attack(agent, obs_tensor, epsilon)
                    elif attack_type == "pgd":
                        obs_tensor = self.pgd_attack(agent, obs_tensor, epsilon)
                    elif attack_type == "random":
                        noise = torch.randn_like(obs_tensor) * epsilon
                        obs_tensor = obs_tensor + noise

                with torch.no_grad():
                    action_out, _, _, _ = agent.ac.get_action_and_value(obs_tensor)
                actions[aid] = int(action_out.item())

            obs_dict, rewards, terms, truncs, _ = env.step(actions)
            total_reward += sum(rewards.values())
            done = any(terms.values()) or any(truncs.values())

        metrics = env.get_metrics()
        coop = getattr(metrics, "cooperation_rate", 0.0)
        gini = getattr(metrics, "gini_coefficient", 0.0)

        return total_reward, coop, gini

    def plot_robustness_curves(
        self,
        profiles: List[RobustnessProfile],
        save_path: Optional[str] = None,
    ):
        """Plot robustness degradation curves for multiple methods."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

        for profile, color in zip(profiles, colors):
            epsilons = [r.epsilon for r in profile.attack_results]
            rewards = [r.attacked_reward for r in profile.attack_results]
            coops = [r.attacked_cooperation for r in profile.attack_results]
            ginis = [r.attacked_gini for r in profile.attack_results]

            axes[0].plot(epsilons, rewards, "o-", color=color,
                        label=profile.method_name, linewidth=2, markersize=6)
            axes[1].plot(epsilons, coops, "o-", color=color,
                        label=profile.method_name, linewidth=2, markersize=6)
            axes[2].plot(epsilons, ginis, "o-", color=color,
                        label=profile.method_name, linewidth=2, markersize=6)

        for ax, title, ylabel in zip(
            axes,
            ["Reward Under Attack", "Cooperation Under Attack", "Gini Under Attack"],
            ["Mean Reward", "Cooperation Rate", "Gini Coefficient"],
        ):
            ax.set_xlabel("Perturbation Budget (epsilon)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Adversarial Robustness Comparison", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
