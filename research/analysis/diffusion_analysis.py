"""
Stigmergic Diffusion Communication Analysis
=============================================
Focused analysis for the diffusion paper: comparing stigmergic
communication against explicit messaging methods.

Key metrics:
- Communication bandwidth (messages/step vs field reads/step)
- Scalability (how metrics change with agent count)
- Information utilization (how much agents use the field)
- Field dynamics (signal persistence, coverage, channel specialization)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


@dataclass
class CommunicationMetrics:
    """Metrics for comparing communication methods."""
    method: str
    num_agents: int
    # Performance
    mean_reward: float = 0.0
    std_reward: float = 0.0
    cooperation_rate: float = 0.0
    gini_coefficient: float = 0.0
    # Communication cost
    messages_per_step: float = 0.0  # explicit: n*(n-1), diffusion: n deposits + n senses
    bytes_per_step: float = 0.0     # approximate bandwidth
    # Timing
    wall_time_per_step_ms: float = 0.0
    # Scalability
    comm_scaling: str = ""  # "O(1)", "O(n)", "O(n^2)"


@dataclass
class DiffusionFieldMetrics:
    """Metrics specific to diffusion field dynamics."""
    # Coverage: fraction of grid cells with non-trivial signal
    field_coverage: float = 0.0
    # Channel utilization: fraction of channels actively used
    channel_utilization: float = 0.0
    # Signal persistence: average signal lifetime before decay
    mean_signal_strength: float = 0.0
    max_signal_strength: float = 0.0
    # Spatial structure: entropy of signal distribution
    spatial_entropy: float = 0.0
    # Channel specialization: do different channels carry different info?
    channel_specialization: float = 0.0
    # Gradient information: are gradients informative?
    mean_gradient_magnitude: float = 0.0


class DiffusionPaperAnalyzer:
    """
    Analysis suite for the stigmergic diffusion paper.

    Computes:
    1. Communication cost comparison (diffusion vs explicit)
    2. Scalability analysis
    3. Diffusion field dynamics
    4. Ablation effect sizes
    """

    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_communication_cost(
        self,
        method: str,
        num_agents: int,
        num_channels: int = 4,
        obs_size: int = 24,
        hidden_size: int = 128,
    ) -> Dict[str, float]:
        """
        Compute theoretical communication cost per timestep.

        Stigmergic diffusion:
        - Each agent deposits to local field: O(1) per agent
        - Each agent senses local field: O(1) per agent
        - Total: O(n) operations, but each is LOCAL (no inter-agent messages)
        - Bandwidth: n * (deposit_size + sense_size) — scales linearly with n
        - But no INTER-AGENT messages — each agent only reads from the shared field

        Explicit messaging (CommNet):
        - Each agent broadcasts hidden state: O(n) messages per agent
        - All agents average: O(n) reads per agent
        - Total: O(n^2) inter-agent messages
        - Bandwidth: n^2 * hidden_size

        Explicit messaging (TarMAC):
        - Each agent sends to all: O(n) messages per agent
        - Attention over received: O(n) reads per agent
        - Total: O(n^2) inter-agent messages
        - Bandwidth: n^2 * (hidden_size + key_size)
        """
        sense_size = 3 * num_channels  # value + grad_x + grad_y per channel
        deposit_size = num_channels     # one strength per channel

        if method == "no_comm":
            return {
                "inter_agent_messages": 0,
                "total_bandwidth_floats": 0,
                "scaling_class": "O(0)",
                "local_operations": 0,
            }
        elif method == "diffusion":
            return {
                "inter_agent_messages": 0,  # No direct inter-agent messages!
                "total_bandwidth_floats": num_agents * (deposit_size + sense_size),
                "scaling_class": "O(n)",
                "local_operations": num_agents * 2,  # deposit + sense per agent
                "field_operations": 1,  # One diffusion step (independent of n)
            }
        elif method == "commnet":
            return {
                "inter_agent_messages": num_agents * (num_agents - 1),
                "total_bandwidth_floats": num_agents * (num_agents - 1) * hidden_size,
                "scaling_class": "O(n^2)",
                "local_operations": num_agents,
            }
        elif method == "tarmac":
            key_size = 16  # typical attention key size
            return {
                "inter_agent_messages": num_agents * (num_agents - 1),
                "total_bandwidth_floats": num_agents * (num_agents - 1) * (hidden_size + key_size),
                "scaling_class": "O(n^2)",
                "local_operations": num_agents,
            }
        elif method == "qmix":
            return {
                "inter_agent_messages": num_agents,  # to central mixer
                "total_bandwidth_floats": num_agents * hidden_size,
                "scaling_class": "O(n)",
                "local_operations": num_agents,
            }
        elif method == "maddpg":
            # Centralized critic sees all obs+actions
            return {
                "inter_agent_messages": num_agents,
                "total_bandwidth_floats": num_agents * (obs_size + 5),  # obs + action
                "scaling_class": "O(n)",
                "local_operations": num_agents,
            }
        else:
            return {"error": f"Unknown method: {method}"}

    def analyze_scalability(
        self,
        results: Dict[str, Dict[str, float]],
        agent_counts: List[int] = None,
    ) -> Dict[str, any]:
        """
        Analyze how performance and cost scale with agent count.

        Args:
            results: {method_n: {reward, cooperation, wall_time, ...}}
            agent_counts: List of agent counts tested

        Returns:
            Scalability analysis including slopes and comparisons
        """
        agent_counts = agent_counts or [4, 8, 16, 32]
        methods = set()
        for key in results:
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                methods.add(parts[0])

        analysis = {}
        for method in methods:
            rewards = []
            times = []
            for n in agent_counts:
                key = f"{method}_{n}"
                if key in results:
                    rewards.append(results[key].get("mean_reward", 0))
                    times.append(results[key].get("wall_time_per_step_ms", 0))

            if len(rewards) >= 2:
                # Compute scaling slope (log-log for power law)
                log_n = np.log(agent_counts[:len(rewards)])
                if any(t > 0 for t in times):
                    log_t = np.log([max(t, 1e-6) for t in times])
                    # Linear fit in log space: log(t) = a * log(n) + b
                    if len(log_n) >= 2:
                        slope, intercept = np.polyfit(log_n, log_t, 1)
                        analysis[method] = {
                            "rewards": rewards,
                            "times": times,
                            "time_scaling_exponent": round(slope, 2),
                            "scaling_class": f"O(n^{slope:.1f})",
                        }
                    else:
                        analysis[method] = {"rewards": rewards, "times": times}
                else:
                    analysis[method] = {"rewards": rewards, "times": times}

            # Add theoretical communication cost
            for n in agent_counts:
                cost = self.compute_communication_cost(method, n)
                analysis.setdefault(method, {}).setdefault("comm_costs", {})[n] = cost

        return analysis

    def analyze_diffusion_field(
        self,
        field_snapshots: List[np.ndarray],
        threshold: float = 0.01,
    ) -> DiffusionFieldMetrics:
        """
        Analyze diffusion field dynamics from recorded snapshots.

        Args:
            field_snapshots: List of field states, each (num_channels, H, W)
            threshold: Minimum signal to count as "active"
        """
        if not field_snapshots:
            return DiffusionFieldMetrics()

        metrics = DiffusionFieldMetrics()

        # Average over snapshots
        all_fields = np.stack(field_snapshots)  # (T, C, H, W)
        mean_field = all_fields.mean(axis=0)    # (C, H, W)

        # Coverage: fraction of cells above threshold
        active_cells = (mean_field > threshold).sum()
        total_cells = mean_field.size
        metrics.field_coverage = float(active_cells / total_cells)

        # Channel utilization
        channel_activity = [(mean_field[c] > threshold).any() for c in range(mean_field.shape[0])]
        metrics.channel_utilization = sum(channel_activity) / len(channel_activity)

        # Signal strength
        metrics.mean_signal_strength = float(mean_field.mean())
        metrics.max_signal_strength = float(mean_field.max())

        # Spatial entropy (per channel, averaged)
        entropies = []
        for c in range(mean_field.shape[0]):
            flat = mean_field[c].flatten()
            total = flat.sum()
            if total > 0:
                probs = flat / total
                probs = probs[probs > 0]
                entropies.append(-np.sum(probs * np.log(probs)))
        metrics.spatial_entropy = float(np.mean(entropies)) if entropies else 0.0

        # Channel specialization (Jensen-Shannon divergence between channels)
        if mean_field.shape[0] >= 2:
            js_divs = []
            for i in range(mean_field.shape[0]):
                for j in range(i + 1, mean_field.shape[0]):
                    p = mean_field[i].flatten()
                    q = mean_field[j].flatten()
                    p_sum, q_sum = p.sum(), q.sum()
                    if p_sum > 0 and q_sum > 0:
                        p_norm = p / p_sum
                        q_norm = q / q_sum
                        m = 0.5 * (p_norm + q_norm)
                        m = np.clip(m, 1e-10, None)
                        kl_pm = np.sum(p_norm[p_norm > 0] * np.log(p_norm[p_norm > 0] / m[p_norm > 0]))
                        kl_qm = np.sum(q_norm[q_norm > 0] * np.log(q_norm[q_norm > 0] / m[q_norm > 0]))
                        js_divs.append(0.5 * (kl_pm + kl_qm))
            metrics.channel_specialization = float(np.mean(js_divs)) if js_divs else 0.0

        # Gradient magnitude
        grad_mags = []
        for c in range(mean_field.shape[0]):
            gy, gx = np.gradient(mean_field[c])
            mag = np.sqrt(gx**2 + gy**2)
            grad_mags.append(mag.mean())
        metrics.mean_gradient_magnitude = float(np.mean(grad_mags))

        return metrics

    def generate_scalability_table(
        self,
        methods: List[str],
        agent_counts: List[int],
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """Generate LaTeX table comparing scalability."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Scalability comparison: reward and communication cost by agent count.}",
            r"\label{tab:scalability}",
            r"\begin{tabular}{l" + "c" * len(agent_counts) * 2 + "}",
            r"\toprule",
        ]

        # Header
        header_parts = ["Method"]
        for n in agent_counts:
            header_parts.append(f"\\multicolumn{{2}}{{c}}{{{n} agents}}")
        lines.append(" & ".join(header_parts) + r" \\")

        sub_header = [""]
        for _ in agent_counts:
            sub_header.extend(["Reward", "Msgs"])
        lines.append(" & ".join(sub_header) + r" \\")
        lines.append(r"\midrule")

        # Data rows
        for method in methods:
            row = [method.replace("_", " ").title()]
            for n in agent_counts:
                key = f"{method}_{n}"
                if key in results:
                    reward = results[key].get("mean_reward", 0)
                    row.append(f"{reward:.1f}")
                else:
                    row.append("--")
                cost = self.compute_communication_cost(method, n)
                row.append(str(cost.get("inter_agent_messages", "--")))
            lines.append(" & ".join(row) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def plot_scalability(
        self,
        methods: List[str],
        agent_counts: List[int],
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ):
        """Plot scalability comparison: reward and comm cost vs agent count."""
        if not PLOT_AVAILABLE:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        colors = {
            "no_comm": "#999999",
            "diffusion": "#2196F3",
            "commnet": "#F44336",
            "tarmac": "#FF9800",
            "qmix": "#4CAF50",
            "maddpg": "#9C27B0",
        }

        # Plot 1: Reward vs agents
        ax = axes[0]
        for method in methods:
            rewards = []
            ns = []
            for n in agent_counts:
                key = f"{method}_{n}"
                if key in results:
                    rewards.append(results[key].get("mean_reward", 0))
                    ns.append(n)
            if ns:
                ax.plot(ns, rewards, 'o-', label=method.replace("_", " ").title(),
                       color=colors.get(method, "#333333"), linewidth=2, markersize=6)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Performance Scalability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Communication cost (theoretical)
        ax = axes[1]
        for method in methods:
            costs = []
            for n in agent_counts:
                cost = self.compute_communication_cost(method, n)
                costs.append(cost.get("inter_agent_messages", 0))
            ax.plot(agent_counts, costs, 'o-', label=method.replace("_", " ").title(),
                   color=colors.get(method, "#333333"), linewidth=2, markersize=6)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Inter-Agent Messages / Step")
        ax.set_title("Communication Cost")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Plot 3: Bandwidth
        ax = axes[2]
        for method in methods:
            bw = []
            for n in agent_counts:
                cost = self.compute_communication_cost(method, n)
                bw.append(cost.get("total_bandwidth_floats", 0))
            ax.plot(agent_counts, bw, 'o-', label=method.replace("_", " ").title(),
                   color=colors.get(method, "#333333"), linewidth=2, markersize=6)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Bandwidth (floats / step)")
        ax.set_title("Bandwidth Scaling")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or str(self.output_dir / "scalability.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_ablation_results(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ):
        """Plot diffusion ablation results (channels, decay, grid, etc.)."""
        if not PLOT_AVAILABLE:
            return

        # Group ablations by type
        groups = {
            "Channels": {},
            "Decay Rate": {},
            "Grid Size": {},
            "Deposit Strength": {},
            "Sense Radius": {},
        }

        for key, result in ablation_results.items():
            reward = result.get("mean_reward", 0)
            if "ch" in key and key.startswith("diffusion_"):
                n = key.replace("diffusion_", "").replace("ch", "")
                groups["Channels"][n] = reward
            elif "decay" in key:
                label = key.replace("diffusion_", "").replace("_decay", "")
                groups["Decay Rate"][label] = reward
            elif "grid" in key:
                size = key.replace("diffusion_grid", "")
                groups["Grid Size"][size] = reward
            elif "deposit" in key:
                label = key.replace("diffusion_", "").replace("_deposit", "")
                groups["Deposit Strength"][label] = reward
            elif "radius" in key:
                r = key.replace("diffusion_radius", "")
                groups["Sense Radius"][r] = reward

        # Filter out empty groups
        groups = {k: v for k, v in groups.items() if v}

        if not groups:
            return

        fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4))
        if len(groups) == 1:
            axes = [axes]

        for ax, (group_name, data) in zip(axes, groups.items()):
            keys = list(data.keys())
            values = list(data.values())
            bars = ax.bar(keys, values, color="#2196F3", alpha=0.8)
            ax.set_title(group_name)
            ax.set_ylabel("Mean Reward")
            ax.grid(True, alpha=0.3, axis='y')

            # Highlight best
            best_idx = np.argmax(values)
            bars[best_idx].set_color("#1565C0")

        plt.tight_layout()
        path = save_path or str(self.output_dir / "ablation_results.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_field_dynamics(
        self,
        field_snapshots: List[np.ndarray],
        timesteps: List[int] = None,
        save_path: Optional[str] = None,
    ):
        """Visualize diffusion field evolution over time."""
        if not PLOT_AVAILABLE or not field_snapshots:
            return

        n_snapshots = min(4, len(field_snapshots))
        indices = np.linspace(0, len(field_snapshots) - 1, n_snapshots, dtype=int)
        num_channels = field_snapshots[0].shape[0]

        fig, axes = plt.subplots(num_channels, n_snapshots,
                                 figsize=(3 * n_snapshots, 3 * num_channels))
        if num_channels == 1:
            axes = axes.reshape(1, -1)
        if n_snapshots == 1:
            axes = axes.reshape(-1, 1)

        channel_names = ["danger", "resource", "coordination", "ethics"]

        for col, idx in enumerate(indices):
            snapshot = field_snapshots[idx]
            t = timesteps[idx] if timesteps else idx
            for c in range(num_channels):
                ax = axes[c, col]
                im = ax.imshow(snapshot[c], cmap='hot', aspect='equal',
                             vmin=0, vmax=max(snapshot[c].max(), 0.01))
                if col == 0:
                    name = channel_names[c] if c < len(channel_names) else f"ch{c}"
                    ax.set_ylabel(name, fontsize=10)
                if c == 0:
                    ax.set_title(f"t={t}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle("Diffusion Field Evolution", fontsize=12)
        plt.tight_layout()
        path = save_path or str(self.output_dir / "field_dynamics.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_paper_summary(
        self,
        main_results: Dict[str, Dict[str, float]],
        ablation_results: Dict[str, Dict[str, float]],
        scalability_results: Dict[str, Dict[str, float]] = None,
        field_metrics: DiffusionFieldMetrics = None,
    ) -> str:
        """Generate a summary of all results for the paper."""
        lines = [
            "# Stigmergic Diffusion Paper — Results Summary",
            "",
            "## Main Comparison (Diffusion vs Baselines)",
            "",
            "| Method | Mean Reward | Std | Cooperation | Gini |",
            "|--------|------------|-----|-------------|------|",
        ]

        for method, res in sorted(main_results.items()):
            lines.append(
                f"| {method} | {res.get('mean_reward', 0):.2f} | "
                f"{res.get('std_reward', 0):.2f} | "
                f"{res.get('cooperation_rate', 0):.3f} | "
                f"{res.get('gini_coefficient', 0):.3f} |"
            )

        if ablation_results:
            lines.extend([
                "",
                "## Diffusion Ablations",
                "",
                "| Config | Mean Reward | Std |",
                "|--------|------------|-----|",
            ])
            for config, res in sorted(ablation_results.items()):
                lines.append(
                    f"| {config} | {res.get('mean_reward', 0):.2f} | "
                    f"{res.get('std_reward', 0):.2f} |"
                )

        if field_metrics:
            lines.extend([
                "",
                "## Diffusion Field Dynamics",
                "",
                f"- Field coverage: {field_metrics.field_coverage:.3f}",
                f"- Channel utilization: {field_metrics.channel_utilization:.3f}",
                f"- Mean signal strength: {field_metrics.mean_signal_strength:.4f}",
                f"- Spatial entropy: {field_metrics.spatial_entropy:.3f}",
                f"- Channel specialization (JS div): {field_metrics.channel_specialization:.4f}",
                f"- Mean gradient magnitude: {field_metrics.mean_gradient_magnitude:.4f}",
            ])

        return "\n".join(lines)
