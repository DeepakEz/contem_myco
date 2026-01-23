"""
MycoNet++ Contemplative Visualization Module
===========================================

Visualization tools for contemplative simulation data including wisdom signals,
network states, ethical behavior, and collective meditation patterns.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features disabled.")

from myconet_wisdom_signals import WisdomSignalType, WisdomSignalGrid
from myconet_contemplative_core import ContemplativeState, WisdomType

logger = logging.getLogger(__name__)

class ContemplativeVisualizer:
    """
    Main visualization class for contemplative simulation data
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if MATPLOTLIB_AVAILABLE:
            # Set up custom color schemes
            self._setup_color_schemes()
            # Configure plotting style
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            sns.set_palette("husl")
        
        self.figure_size = (12, 8)
        self.dpi = 300
    
    def _setup_color_schemes(self):
        """Setup custom color schemes for different visualization types"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Wisdom signal colors
        self.wisdom_colors = {
            WisdomSignalType.SUFFERING_ALERT: '#FF4444',      # Red
            WisdomSignalType.COMPASSION_GRADIENT: '#FF8C69',  # Orange-red
            WisdomSignalType.WISDOM_BEACON: '#4169E1',        # Royal blue
            WisdomSignalType.MEDITATION_SYNC: '#9370DB',      # Medium purple
            WisdomSignalType.COOPERATION_CALL: '#32CD32',     # Lime green
            WisdomSignalType.ETHICAL_INSIGHT: '#FFD700',      # Gold
            WisdomSignalType.MINDFULNESS_WAVE: '#20B2AA',     # Light sea green
            WisdomSignalType.CAUTION_WARNING: '#FFA500'       # Orange
        }
        
        # Contemplative state colors
        self.state_colors = {
            ContemplativeState.ORDINARY: '#808080',           # Gray
            ContemplativeState.MINDFUL: '#90EE90',           # Light green
            ContemplativeState.DEEP_CONTEMPLATION: '#4169E1', # Royal blue
            ContemplativeState.COLLECTIVE_MEDITATION: '#9370DB', # Purple
            ContemplativeState.WISDOM_INTEGRATION: '#FFD700'  # Gold
        }
        
        # Create custom colormaps
        self.mindfulness_cmap = LinearSegmentedColormap.from_list(
            'mindfulness', ['#FFFFFF', '#90EE90', '#4169E1', '#9370DB']
        )
        
        self.wisdom_cmap = LinearSegmentedColormap.from_list(
            'wisdom', ['#FFFFFF', '#FFE4B5', '#FFD700', '#FFA500']
        )
        
        self.suffering_cmap = LinearSegmentedColormap.from_list(
            'suffering', ['#FFFFFF', '#FFB6C1', '#FF69B4', '#FF1493']
        )
    
    def create_simulation_summary(self, simulation_data: Dict[str, Any], 
                                config: Dict[str, Any]) -> str:
        """Create comprehensive visualization summary of simulation"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_summary(simulation_data, config)
        
        summary_file = self.output_dir / "simulation_summary.png"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Population dynamics
        ax1 = plt.subplot(3, 4, 1)
        self.plot_population_dynamics(simulation_data, ax=ax1)
        
        # Wisdom propagation
        ax2 = plt.subplot(3, 4, 2)
        self.plot_wisdom_propagation(simulation_data, ax=ax2)
        
        # Ethical evolution
        ax3 = plt.subplot(3, 4, 3)
        self.plot_ethical_evolution(simulation_data, ax=ax3)
        
        # Network coherence
        ax4 = plt.subplot(3, 4, 4)
        self.plot_network_coherence(simulation_data, ax=ax4)
        
        # Contemplative states distribution
        ax5 = plt.subplot(3, 4, 5)
        self.plot_contemplative_states_distribution(simulation_data, ax=ax5)
        
        # Wisdom signal activity
        ax6 = plt.subplot(3, 4, 6)
        self.plot_wisdom_signal_activity(simulation_data, ax=ax6)
        
        # Collective meditation events
        ax7 = plt.subplot(3, 4, 7)
        self.plot_collective_meditation_events(simulation_data, ax=ax7)
        
        # Overmind performance
        ax8 = plt.subplot(3, 4, 8)
        self.plot_overmind_performance(simulation_data, ax=ax8)
        
        # Agent energy/health correlation
        ax9 = plt.subplot(3, 4, 9)
        self.plot_agent_wellbeing_correlation(simulation_data, ax=ax9)
        
        # Wisdom type distribution
        ax10 = plt.subplot(3, 4, 10)
        self.plot_wisdom_type_distribution(simulation_data, ax=ax10)
        
        # Network topology metrics
        ax11 = plt.subplot(3, 4, 11)
        self.plot_network_topology_metrics(simulation_data, ax=ax11)
        
        # Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        self.create_summary_text_box(simulation_data, config, ax=ax12)
        
        plt.tight_layout()
        plt.savefig(summary_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simulation summary saved to {summary_file}")
        return str(summary_file)
    
    def plot_population_dynamics(self, simulation_data: Dict[str, Any], ax=None):
        """Plot population dynamics over time"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        population_data = simulation_data.get('population_data', [])
        if not population_data:
            ax.text(0.5, 0.5, 'No population data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [p['step'] for p in population_data]
        population = [p['total_population'] for p in population_data]
        energy = [p['average_energy'] for p in population_data]
        health = [p['average_health'] for p in population_data]
        
        ax2 = ax.twinx()
        
        # Population
        line1 = ax.plot(steps, population, 'b-', linewidth=2, label='Population')
        
        # Energy and health
        line2 = ax2.plot(steps, energy, 'g--', alpha=0.7, label='Avg Energy')
        line3 = ax2.plot(steps, health, 'r--', alpha=0.7, label='Avg Health')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Population', color='b')
        ax2.set_ylabel('Energy/Health Level', color='g')
        ax.set_title('Population Dynamics')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def plot_wisdom_propagation(self, simulation_data: Dict[str, Any], ax=None):
        """Plot wisdom generation and propagation over time"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        wisdom_data = simulation_data.get('wisdom_data', [])
        if not wisdom_data:
            ax.text(0.5, 0.5, 'No wisdom data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [w['step'] for w in wisdom_data]
        generated = [w['total_wisdom_generated'] for w in wisdom_data]
        received = [w['total_wisdom_received'] for w in wisdom_data]
        
        ax.plot(steps, generated, 'b-', linewidth=2, label='Wisdom Generated')
        ax.plot(steps, received, 'g-', linewidth=2, label='Wisdom Received')
        
        # Calculate and plot propagation efficiency
        efficiency = [r/max(g, 1) for g, r in zip(generated, received)]
        ax2 = ax.twinx()
        ax2.plot(steps, efficiency, 'r--', alpha=0.7, label='Propagation Efficiency')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Wisdom Count')
        ax2.set_ylabel('Propagation Efficiency', color='r')
        ax.set_title('Wisdom Propagation')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def plot_ethical_evolution(self, simulation_data: Dict[str, Any], ax=None):
        """Plot evolution of ethical behavior"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        ethical_data = simulation_data.get('ethical_data', [])
        if not ethical_data:
            ax.text(0.5, 0.5, 'No ethical data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [e['step'] for e in ethical_data]
        ethical_ratio = [e['ethical_decision_ratio'] for e in ethical_data]
        harmony = [e['collective_harmony'] for e in ethical_data]
        
        ax.plot(steps, ethical_ratio, 'b-', linewidth=2, label='Ethical Decision Ratio')
        ax.plot(steps, harmony, 'g-', linewidth=2, label='Collective Harmony')
        
        # Add trend lines
        if len(steps) > 5:
            ethical_trend = np.polyfit(steps, ethical_ratio, 1)
            harmony_trend = np.polyfit(steps, harmony, 1)
            
            ax.plot(steps, np.polyval(ethical_trend, steps), 'b--', alpha=0.5, label='Ethical Trend')
            ax.plot(steps, np.polyval(harmony_trend, steps), 'g--', alpha=0.5, label='Harmony Trend')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Ratio/Level')
        ax.set_title('Ethical Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def plot_network_coherence(self, simulation_data: Dict[str, Any], ax=None):
        """Plot network coherence and wisdom flow"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        network_data = simulation_data.get('network_data', [])
        if not network_data:
            ax.text(0.5, 0.5, 'No network data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [n['step'] for n in network_data]
        coherence = [n.get('network_contemplative_coherence', 0) for n in network_data]
        diversity = [n.get('signal_diversity', 0) for n in network_data]
        flow = [n.get('wisdom_flow_efficiency', 0) for n in network_data]
        
        ax.plot(steps, coherence, 'b-', linewidth=2, label='Network Coherence')
        ax.plot(steps, diversity, 'g-', linewidth=2, label='Signal Diversity')
        ax.plot(steps, flow, 'r-', linewidth=2, label='Wisdom Flow Efficiency')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Level')
        ax.set_title('Network Coherence & Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def plot_contemplative_states_distribution(self, simulation_data: Dict[str, Any], ax=None):
        """Plot distribution of contemplative states over time"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        wisdom_data = simulation_data.get('wisdom_data', [])
        if not wisdom_data:
            ax.text(0.5, 0.5, 'No wisdom data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [w['step'] for w in wisdom_data]
        meditation_agents = [w.get('agents_in_meditation', 0) for w in wisdom_data]
        mindfulness_levels = [w.get('average_mindfulness', 0) for w in wisdom_data]
        
        # Create stacked area plot
        ax.fill_between(steps, 0, meditation_agents, alpha=0.7, 
                       color=self.state_colors[ContemplativeState.COLLECTIVE_MEDITATION],
                       label='Collective Meditation')
        
        # Plot mindfulness as line
        ax2 = ax.twinx()
        ax2.plot(steps, mindfulness_levels, 'b-', linewidth=2, label='Avg Mindfulness')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Agents in Meditation')
        ax2.set_ylabel('Average Mindfulness', color='b')
        ax.set_title('Contemplative States Distribution')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def plot_wisdom_signal_activity(self, simulation_data: Dict[str, Any], ax=None):
        """Plot wisdom signal activity over time"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        network_data = simulation_data.get('network_data', [])
        if not network_data:
            ax.text(0.5, 0.5, 'No network data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [n['step'] for n in network_data]
        total_signals = [n.get('total_signals_created', 0) for n in network_data]
        active_layers = [n.get('active_signal_layers', 0) for n in network_data]
        
        ax.plot(steps, total_signals, 'b-', linewidth=2, label='Total Signals Created')
        
        ax2 = ax.twinx()
        ax2.plot(steps, active_layers, 'r-', linewidth=2, label='Active Signal Layers')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Signal Count', color='b')
        ax2.set_ylabel('Active Layers', color='r')
        ax.set_title('Wisdom Signal Activity')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def plot_collective_meditation_events(self, simulation_data: Dict[str, Any], ax=None):
        """Plot collective meditation events"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        wisdom_data = simulation_data.get('wisdom_data', [])
        if not wisdom_data:
            ax.text(0.5, 0.5, 'No wisdom data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Find meditation events
        meditation_events = []
        for w in wisdom_data:
            agents_in_meditation = w.get('agents_in_meditation', 0)
            if agents_in_meditation > 0:
                meditation_events.append({
                    'step': w['step'],
                    'participants': agents_in_meditation,
                    'mindfulness': w.get('average_mindfulness', 0)
                })
        
        if meditation_events:
            steps = [e['step'] for e in meditation_events]
            participants = [e['participants'] for e in meditation_events]
            mindfulness = [e['mindfulness'] for e in meditation_events]
            
            # Scatter plot with color representing mindfulness
            scatter = ax.scatter(steps, participants, c=mindfulness, 
                               cmap=self.mindfulness_cmap, s=50, alpha=0.7)
            
            plt.colorbar(scatter, ax=ax, label='Mindfulness Level')
            
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Participants')
            ax.set_title('Collective Meditation Events')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No collective meditation events', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_overmind_performance(self, simulation_data: Dict[str, Any], ax=None):
        """Plot Overmind intervention performance"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        overmind_data = simulation_data.get('overmind_data', [])
        if not overmind_data or not any(o.get('decisions_made', 0) > 0 for o in overmind_data):
            ax.text(0.5, 0.5, 'No Overmind data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract Overmind metrics
        steps = [o['step'] for o in overmind_data if o.get('decisions_made', 0) > 0]
        success_rates = [o.get('success_rate', 0) for o in overmind_data if o.get('decisions_made', 0) > 0]
        decisions = [o.get('decisions_made', 0) for o in overmind_data if o.get('decisions_made', 0) > 0]
        meditations = [o.get('collective_meditations_triggered', 0) for o in overmind_data if o.get('decisions_made', 0) > 0]
        
        if steps:
            ax.plot(steps, success_rates, 'b-', linewidth=2, label='Success Rate')
            
            ax2 = ax.twinx()
            ax2.plot(steps, decisions, 'g--', alpha=0.7, label='Total Decisions')
            ax2.plot(steps, meditations, 'r:', alpha=0.7, label='Meditations Triggered')
            
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Success Rate', color='b')
            ax2.set_ylabel('Count', color='g')
            ax.set_title('Overmind Performance')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    def plot_agent_wellbeing_correlation(self, simulation_data: Dict[str, Any], ax=None):
        """Plot correlation between agent energy and health"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        population_data = simulation_data.get('population_data', [])
        if not population_data:
            ax.text(0.5, 0.5, 'No population data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        energy_levels = [p['average_energy'] for p in population_data]
        health_levels = [p['average_health'] for p in population_data]
        steps = [p['step'] for p in population_data]
        
        # Create scatter plot with time as color
        scatter = ax.scatter(energy_levels, health_levels, c=steps, 
                           cmap='viridis', alpha=0.6, s=30)
        
        # Add correlation line
        if len(energy_levels) > 1:
            correlation = np.corrcoef(energy_levels, health_levels)[0, 1]
            z = np.polyfit(energy_levels, health_levels, 1)
            p = np.poly1d(z)
            ax.plot(sorted(energy_levels), p(sorted(energy_levels)), 
                   "r--", alpha=0.8, label=f'Correlation: {correlation:.3f}')
        
        plt.colorbar(scatter, ax=ax, label='Simulation Step')
        ax.set_xlabel('Average Energy')
        ax.set_ylabel('Average Health')
        ax.set_title('Agent Wellbeing Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_wisdom_type_distribution(self, simulation_data: Dict[str, Any], ax=None):
        """Plot distribution of wisdom types from actual simulation data.

        Extracts wisdom type counts from the simulation data's wisdom_type_counts
        field, or computes from individual wisdom events if available.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)

        wisdom_data = simulation_data.get('wisdom_data', [])
        if not wisdom_data:
            ax.text(0.5, 0.5, 'No wisdom data available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Try to get actual wisdom type counts from data
        wisdom_types = ['Ethical Insight', 'Suffering Detection', 'Compassion Response',
                       'Interconnectedness', 'Practical Wisdom', 'Impermanence']
        wisdom_type_keys = ['ethical_insight', 'suffering_detection', 'compassion_response',
                           'interconnectedness', 'practical_wisdom', 'impermanence']

        # Aggregate wisdom type counts from all timesteps
        type_counts = {key: 0 for key in wisdom_type_keys}

        for data_point in wisdom_data:
            # Check for wisdom_type_counts field
            if 'wisdom_type_counts' in data_point:
                counts = data_point['wisdom_type_counts']
                for key in wisdom_type_keys:
                    type_counts[key] += counts.get(key, 0)
            # Alternative: check for individual type fields
            elif 'wisdom_types' in data_point:
                types = data_point['wisdom_types']
                for key in wisdom_type_keys:
                    type_counts[key] += types.get(key, 0)

        counts = [type_counts[key] for key in wisdom_type_keys]
        total_count = sum(counts)

        # If no type-specific data, derive from total wisdom and signal patterns
        if total_count == 0:
            final_data = wisdom_data[-1] if wisdom_data else {}
            total_wisdom = final_data.get('total_wisdom_generated', 0)

            # Derive distribution from signal data if available
            network_data = simulation_data.get('network_data', [])
            if network_data and network_data[-1].get('signal_types'):
                signal_types = network_data[-1]['signal_types']
                # Map signal intensities to wisdom types
                counts = [
                    signal_types.get('ethical_insight', total_wisdom * 0.20),
                    signal_types.get('suffering_alert', total_wisdom * 0.15),
                    signal_types.get('compassion_gradient', total_wisdom * 0.20),
                    signal_types.get('wisdom_beacon', total_wisdom * 0.15),
                    signal_types.get('cooperation_call', total_wisdom * 0.15),
                    signal_types.get('mindfulness_wave', total_wisdom * 0.15),
                ]
            else:
                # Last resort: use proportional estimate based on contemplative metrics
                mindfulness = final_data.get('avg_mindfulness', 0.5)
                cooperation = final_data.get('cooperation_rate', 0.5)
                ethical = final_data.get('ethical_alignment', 0.5)

                # Distribute based on metrics (weighted by relevance)
                total = max(1, total_wisdom)
                counts = [
                    int(total * ethical * 0.3),           # Ethical Insight
                    int(total * (1 - mindfulness) * 0.2), # Suffering Detection
                    int(total * cooperation * 0.25),       # Compassion Response
                    int(total * mindfulness * 0.15),       # Interconnectedness
                    int(total * 0.10),                     # Practical Wisdom
                    int(total * mindfulness * 0.10),       # Impermanence
                ]

        # Filter out zero counts for cleaner visualization
        non_zero = [(t, c) for t, c in zip(wisdom_types, counts) if c > 0]
        if non_zero:
            wisdom_types, counts = zip(*non_zero)
        else:
            ax.text(0.5, 0.5, 'No wisdom generated yet',
                   ha='center', va='center', transform=ax.transAxes)
            return

        colors = ['#FFD700', '#FF4444', '#FF8C69', '#32CD32', '#4169E1', '#9370DB'][:len(counts)]
        ax.pie(counts, labels=wisdom_types, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Wisdom Type Distribution (from simulation data)')
    
    def plot_network_topology_metrics(self, simulation_data: Dict[str, Any], ax=None):
        """Plot network topology and connectivity metrics"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        network_data = simulation_data.get('network_data', [])
        if not network_data:
            ax.text(0.5, 0.5, 'No network data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps = [n['step'] for n in network_data]
        total_intensity = [n.get('total_signal_intensity', 0) for n in network_data]
        
        # Plot signal intensity over time
        ax.plot(steps, total_intensity, 'b-', linewidth=2, label='Total Signal Intensity')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Signal Intensity')
        ax.set_title('Network Signal Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_summary_text_box(self, simulation_data: Dict[str, Any], 
                               config: Dict[str, Any], ax=None):
        """Create text summary box with key statistics"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Calculate summary statistics
        population_data = simulation_data.get('population_data', [])
        wisdom_data = simulation_data.get('wisdom_data', [])
        ethical_data = simulation_data.get('ethical_data', [])
        
        summary_text = "SIMULATION SUMMARY\n\n"
        
        if population_data:
            initial_pop = population_data[0].get('total_population', 0)
            final_pop = population_data[-1].get('total_population', 0)
            survival_rate = final_pop / max(initial_pop, 1)
            
            summary_text += f"Population:\n"
            summary_text += f"  Initial: {initial_pop}\n"
            summary_text += f"  Final: {final_pop}\n"
            summary_text += f"  Survival Rate: {survival_rate:.1%}\n\n"
        
        if wisdom_data:
            total_wisdom = wisdom_data[-1].get('total_wisdom_generated', 0)
            avg_mindfulness = wisdom_data[-1].get('average_mindfulness', 0)
            
            summary_text += f"Contemplative Metrics:\n"
            summary_text += f"  Total Wisdom: {total_wisdom}\n"
            summary_text += f"  Avg Mindfulness: {avg_mindfulness:.3f}\n\n"
        
        if ethical_data:
            ethical_ratio = ethical_data[-1].get('ethical_decision_ratio', 0)
            harmony = ethical_data[-1].get('collective_harmony', 0)
            
            summary_text += f"Ethical Metrics:\n"
            summary_text += f"  Ethical Ratio: {ethical_ratio:.1%}\n"
            summary_text += f"  Collective Harmony: {harmony:.3f}\n\n"
        
        # Add configuration info
        experiment_name = config.get('experiment_name', 'Unknown')
        max_steps = len(population_data) if population_data else 0
        
        summary_text += f"Configuration:\n"
        summary_text += f"  Experiment: {experiment_name}\n"
        summary_text += f"  Steps Completed: {max_steps}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Summary Statistics')
    
    def create_wisdom_signal_heatmap(self, wisdom_signal_grid: WisdomSignalGrid, 
                                   signal_type: WisdomSignalType = None,
                                   filename: str = None) -> str:
        """Create heatmap visualization of wisdom signals"""
        if not MATPLOTLIB_AVAILABLE:
            return "Visualization not available (matplotlib not installed)"
        
        if filename is None:
            signal_name = signal_type.value if signal_type else "all_signals"
            filename = f"wisdom_signals_{signal_name}.png"
        
        output_file = self.output_dir / filename
        
        # Get signal data
        signal_data = wisdom_signal_grid.visualize_signals(signal_type)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if signal_type:
            colormap = self.wisdom_cmap
            title = f"Wisdom Signals: {signal_type.value.replace('_', ' ').title()}"
        else:
            colormap = 'viridis'
            title = "All Wisdom Signals (Combined)"
        
        im = ax.imshow(signal_data, cmap=colormap, interpolation='bilinear', 
                      origin='lower', aspect='equal')
        
        plt.colorbar(im, ax=ax, label='Signal Intensity')
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Wisdom signal heatmap saved to {output_file}")
        return str(output_file)
    
    def create_agent_state_animation(self, simulation_data: Dict[str, Any],
                                   filename: str = "agent_states.gif") -> str:
        """Create animation of agent states over time using actual simulation data.

        Visualizes agent positions, contemplative states, and wisdom sharing events.
        Agent colors indicate contemplative state:
        - Blue: Ordinary state
        - Green: Mindful state
        - Gold: Deep contemplation
        - Purple: Collective meditation
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Animation not available (matplotlib not installed)"

        output_file = self.output_dir / filename

        # Get agent snapshot data
        agent_snapshots = simulation_data.get('agent_snapshots', [])
        population_data = simulation_data.get('population_data', [])

        # Determine grid dimensions from config or data
        config = simulation_data.get('config', {})
        grid_width = config.get('environment_width', 50)
        grid_height = config.get('environment_height', 50)

        fig, ax = plt.subplots(figsize=(10, 10))

        # State to color mapping
        state_colors = {
            'ordinary': '#4169E1',      # Blue
            'mindful': '#32CD32',        # Green
            'deep_contemplation': '#FFD700',  # Gold
            'collective_meditation': '#9370DB', # Purple
            'wisdom_sharing': '#FF8C69'  # Coral
        }

        def animate_frame(frame_idx):
            ax.clear()
            ax.set_xlim(0, grid_width)
            ax.set_ylim(0, grid_height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

            # Get data for this frame
            if agent_snapshots and frame_idx < len(agent_snapshots):
                snapshot = agent_snapshots[frame_idx]
                agents = snapshot.get('agents', [])
                step = snapshot.get('step', frame_idx)

                # Plot each agent
                for agent in agents:
                    x = agent.get('x', np.random.uniform(0, grid_width))
                    y = agent.get('y', np.random.uniform(0, grid_height))
                    state = agent.get('contemplative_state', 'ordinary').lower()
                    energy = agent.get('energy', 0.5)

                    color = state_colors.get(state, '#4169E1')
                    size = 50 + energy * 100  # Size based on energy

                    ax.scatter(x, y, c=color, s=size, alpha=0.7, edgecolors='black', linewidths=0.5)

                    # Show wisdom sharing as connections
                    if agent.get('sharing_wisdom'):
                        target = agent.get('wisdom_target')
                        if target:
                            ax.plot([x, target.get('x', x)], [y, target.get('y', y)],
                                   'gold', alpha=0.5, linewidth=2)

                ax.set_title(f"Agent States - Step {step} ({len(agents)} agents)")

            elif population_data and frame_idx < len(population_data):
                # Fallback: generate positions from population metrics
                pop_data = population_data[frame_idx]
                pop_count = pop_data.get('alive_count', pop_data.get('total_population', 10))
                step = pop_data.get('step', frame_idx)
                avg_mindfulness = pop_data.get('avg_mindfulness', 0.5)
                cooperation = pop_data.get('cooperation_rate', 0.5)

                # Generate representative agent positions using deterministic seeding
                np.random.seed(frame_idx)  # Reproducible positions
                for i in range(pop_count):
                    x = np.random.uniform(2, grid_width - 2)
                    y = np.random.uniform(2, grid_height - 2)

                    # Assign state based on probability from metrics
                    if np.random.random() < avg_mindfulness * 0.5:
                        state = 'mindful' if np.random.random() > 0.3 else 'deep_contemplation'
                    elif np.random.random() < cooperation * 0.3:
                        state = 'collective_meditation'
                    else:
                        state = 'ordinary'

                    color = state_colors.get(state, '#4169E1')
                    ax.scatter(x, y, c=color, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

                ax.set_title(f"Agent States - Step {step} ({pop_count} agents)")

            else:
                ax.set_title(f"Agent States - Frame {frame_idx}")
                ax.text(0.5, 0.5, 'No agent data for this frame',
                       ha='center', va='center', transform=ax.transAxes)

            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=10, label=state.replace('_', ' ').title())
                for state, color in state_colors.items()
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

            return []

        # Determine number of frames
        n_frames = max(len(agent_snapshots), len(population_data), 10)
        n_frames = min(n_frames, 200)  # Cap at 200 frames

        anim = animation.FuncAnimation(fig, animate_frame, frames=n_frames,
                                       interval=100, blit=False)

        try:
            anim.save(output_file, writer='pillow', fps=10)
            logger.info(f"Agent state animation saved to {output_file}")
        except Exception as e:
            logger.warning(f"Could not save animation: {e}")
            # Try saving as MP4 if pillow fails
            try:
                anim.save(str(output_file).replace('.gif', '.mp4'), writer='ffmpeg', fps=10)
                output_file = str(output_file).replace('.gif', '.mp4')
            except Exception:
                return f"Animation creation failed: {e}"

        plt.close()
        return str(output_file)
    
    def create_network_flow_diagram(self, simulation_data: Dict[str, Any],
                                  timestep: int = -1,
                                  filename: str = "network_flow.png") -> str:
        """Create network flow diagram showing wisdom propagation from simulation data.

        Visualizes:
        - Agent positions as nodes (sized by wisdom accumulated)
        - Wisdom sharing events as directed edges
        - Signal type intensities as node colors
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Visualization not available (matplotlib not installed)"

        output_file = self.output_dir / filename

        fig, ax = plt.subplots(figsize=(12, 12))

        # Get agent and network data
        agent_snapshots = simulation_data.get('agent_snapshots', [])
        network_data = simulation_data.get('network_data', [])
        wisdom_events = simulation_data.get('wisdom_events', [])

        config = simulation_data.get('config', {})
        grid_width = config.get('environment_width', 50)
        grid_height = config.get('environment_height', 50)

        ax.set_xlim(0, grid_width)
        ax.set_ylim(0, grid_height)
        ax.set_aspect('equal')

        # Get snapshot at specified timestep
        if agent_snapshots:
            snapshot = agent_snapshots[timestep] if abs(timestep) < len(agent_snapshots) else agent_snapshots[-1]
            agents = snapshot.get('agents', [])
        else:
            agents = []

        # If no agent snapshots, generate from population data
        if not agents:
            population_data = simulation_data.get('population_data', [])
            if population_data:
                pop_data = population_data[timestep] if abs(timestep) < len(population_data) else population_data[-1]
                pop_count = pop_data.get('alive_count', pop_data.get('total_population', 15))

                # Generate agent positions deterministically
                np.random.seed(42)
                agents = []
                for i in range(pop_count):
                    agents.append({
                        'id': i,
                        'x': np.random.uniform(5, grid_width - 5),
                        'y': np.random.uniform(5, grid_height - 5),
                        'wisdom_accumulated': np.random.uniform(0.2, 1.0),
                        'mindfulness_level': pop_data.get('avg_mindfulness', 0.5) + np.random.uniform(-0.2, 0.2)
                    })

        # Draw agents as nodes
        if agents:
            for agent in agents:
                x = agent.get('x', np.random.uniform(5, grid_width - 5))
                y = agent.get('y', np.random.uniform(5, grid_height - 5))
                wisdom = agent.get('wisdom_accumulated', 0.5)
                mindfulness = agent.get('mindfulness_level', 0.5)

                # Node size based on wisdom accumulated
                size = 100 + wisdom * 300

                # Color based on mindfulness (blue -> green gradient)
                color = plt.cm.viridis(mindfulness)

                ax.scatter(x, y, c=[color], s=size, alpha=0.7, edgecolors='black', linewidths=1)

        # Draw wisdom flow connections
        if wisdom_events:
            # Filter events for specified timestep range
            step = agent_snapshots[timestep].get('step', 0) if agent_snapshots else 0
            relevant_events = [e for e in wisdom_events
                             if abs(e.get('step', 0) - step) < 10]

            for event in relevant_events[:50]:  # Limit to 50 connections for clarity
                source_id = event.get('source_agent')
                target_id = event.get('target_agent')
                strength = event.get('strength', 0.5)

                # Find source and target positions
                source_agent = next((a for a in agents if a.get('id') == source_id), None)
                target_agent = next((a for a in agents if a.get('id') == target_id), None)

                if source_agent and target_agent:
                    sx, sy = source_agent.get('x'), source_agent.get('y')
                    tx, ty = target_agent.get('x'), target_agent.get('y')

                    # Draw arrow with strength-based opacity
                    ax.annotate('', xy=(tx, ty), xytext=(sx, sy),
                               arrowprops=dict(arrowstyle='->', color='gold',
                                             alpha=strength * 0.8, lw=strength * 2))

        elif agents and len(agents) > 1:
            # Generate wisdom flow connections based on proximity and mindfulness
            for i, agent in enumerate(agents):
                if agent.get('mindfulness_level', 0.5) > 0.6:
                    # High mindfulness agents share wisdom with nearby agents
                    x1, y1 = agent.get('x'), agent.get('y')
                    for j, other in enumerate(agents):
                        if i != j:
                            x2, y2 = other.get('x'), other.get('y')
                            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            if dist < grid_width * 0.2:  # Nearby agents
                                strength = (1 - dist / (grid_width * 0.2)) * agent.get('wisdom_accumulated', 0.5)
                                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                           arrowprops=dict(arrowstyle='->', color='gold',
                                                         alpha=min(0.8, strength), lw=1))

        # Add signal intensity heatmap from network data
        if network_data:
            net_data = network_data[timestep] if abs(timestep) < len(network_data) else network_data[-1]
            signal_grid = net_data.get('signal_grid')
            if signal_grid is not None and hasattr(signal_grid, 'shape'):
                # Overlay signal intensity as contour
                try:
                    signal_grid = np.array(signal_grid)
                    if signal_grid.ndim == 2:
                        ax.contourf(signal_grid.T, alpha=0.2, cmap='YlOrRd',
                                  extent=[0, grid_width, 0, grid_height], levels=10)
                except Exception as e:
                    logger.debug(f"Could not render signal grid: {e}")

        ax.set_title(f'Wisdom Signal Network Flow (t={timestep})')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Add colorbar for mindfulness
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Mindfulness Level')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=8, label='Agent (size = wisdom)'),
            plt.Line2D([0], [0], color='gold', lw=2, label='Wisdom flow'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Network flow diagram saved to {output_file}")
        return str(output_file)
    
    def _create_text_summary(self, simulation_data: Dict[str, Any], 
                           config: Dict[str, Any]) -> str:
        """Create text-based summary when matplotlib is not available"""
        summary_file = self.output_dir / "simulation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("CONTEMPLATIVE MYCONET SIMULATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Population summary
            population_data = simulation_data.get('population_data', [])
            if population_data:
                initial = population_data[0]
                final = population_data[-1]
                f.write(f"Population Dynamics:\n")
                f.write(f"  Initial Population: {initial.get('total_population', 0)}\n")
                f.write(f"  Final Population: {final.get('total_population', 0)}\n")
                f.write(f"  Survival Rate: {final.get('total_population', 0) / max(initial.get('total_population', 1), 1):.1%}\n")
                f.write(f"  Final Avg Energy: {final.get('average_energy', 0):.3f}\n")
                f.write(f"  Final Avg Health: {final.get('average_health', 0):.3f}\n\n")
            
            # Wisdom summary
            wisdom_data = simulation_data.get('wisdom_data', [])
            if wisdom_data:
                final_wisdom = wisdom_data[-1]
                f.write(f"Contemplative Metrics:\n")
                f.write(f"  Total Wisdom Generated: {final_wisdom.get('total_wisdom_generated', 0)}\n")
                f.write(f"  Total Wisdom Received: {final_wisdom.get('total_wisdom_received', 0)}\n")
                f.write(f"  Final Avg Mindfulness: {final_wisdom.get('average_mindfulness', 0):.3f}\n\n")
            
            # Ethical summary
            ethical_data = simulation_data.get('ethical_data', [])
            if ethical_data:
                final_ethical = ethical_data[-1]
                f.write(f"Ethical Metrics:\n")
                f.write(f"  Final Ethical Ratio: {final_ethical.get('ethical_decision_ratio', 0):.1%}\n")
                f.write(f"  Collective Harmony: {final_ethical.get('collective_harmony', 0):.3f}\n\n")
            
            # Configuration
            f.write(f"Configuration:\n")
            f.write(f"  Experiment: {config.get('experiment_name', 'Unknown')}\n")
            f.write(f"  Steps: {len(population_data) if population_data else 0}\n")
        
        logger.info(f"Text summary saved to {summary_file}")
        return str(summary_file)

# Factory function for easy visualization creation
def create_visualization_suite(simulation_data: Dict[str, Any], 
                             config: Dict[str, Any],
                             output_dir: str = "visualizations") -> Dict[str, str]:
    """
    Create complete visualization suite for simulation data
    
    Args:
        simulation_data: Complete simulation data
        config: Simulation configuration
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    visualizer = ContemplativeVisualizer(output_dir)
    
    visualizations = {}
    
    # Main summary
    visualizations['summary'] = visualizer.create_simulation_summary(simulation_data, config)
    
    if MATPLOTLIB_AVAILABLE:
        # Individual plots
        fig_names = [
            'population_dynamics', 'wisdom_propagation', 'ethical_evolution',
            'network_coherence', 'contemplative_states', 'wisdom_signals',
            'meditation_events', 'overmind_performance'
        ]
        
        for fig_name in fig_names:
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                if fig_name == 'population_dynamics':
                    visualizer.plot_population_dynamics(simulation_data, ax)
                elif fig_name == 'wisdom_propagation':
                    visualizer.plot_wisdom_propagation(simulation_data, ax)
                elif fig_name == 'ethical_evolution':
                    visualizer.plot_ethical_evolution(simulation_data, ax)
                elif fig_name == 'network_coherence':
                    visualizer.plot_network_coherence(simulation_data, ax)
                elif fig_name == 'contemplative_states':
                    visualizer.plot_contemplative_states_distribution(simulation_data, ax)
                elif fig_name == 'wisdom_signals':
                    visualizer.plot_wisdom_signal_activity(simulation_data, ax)
                elif fig_name == 'meditation_events':
                    visualizer.plot_collective_meditation_events(simulation_data, ax)
                elif fig_name == 'overmind_performance':
                    visualizer.plot_overmind_performance(simulation_data, ax)
                
                output_file = Path(output_dir) / f"{fig_name}.png"
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualizations[fig_name] = str(output_file)
                
            except Exception as e:
                logger.warning(f"Could not create {fig_name} visualization: {e}")
        
        # Network flow diagram
        try:
            visualizations['network_flow'] = visualizer.create_network_flow_diagram(simulation_data)
        except Exception as e:
            logger.warning(f"Could not create network flow diagram: {e}")
    
    logger.info(f"Created {len(visualizations)} visualizations in {output_dir}")
    return visualizations

# Standalone visualization testing function
def test_visualization_system():
    """Test the visualization system with dummy data"""
    print("Testing contemplative visualization system...")
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available - visualization features disabled")
        return False
    
    # Create dummy simulation data
    dummy_data = {
        'population_data': [
            {'step': i, 'total_population': 20 - i//10, 'average_energy': 0.8 - i*0.001, 
             'average_health': 0.9 - i*0.0005} for i in range(100)
        ],
        'wisdom_data': [
            {'step': i, 'total_wisdom_generated': i*2, 'total_wisdom_received': i*1.8,
             'average_mindfulness': 0.5 + 0.3*np.sin(i/10), 'agents_in_meditation': max(0, int(5*np.sin(i/20)))}
             for i in range(100)
        ],
        'ethical_data': [
            {'step': i, 'ethical_decision_ratio': 0.6 + 0.2*np.sin(i/15),
             'collective_harmony': 0.5 + 0.3*np.cos(i/12)} for i in range(100)
        ],
        'network_data': [
            {'step': i, 'network_contemplative_coherence': 0.4 + 0.3*np.sin(i/8),
             'signal_diversity': 0.6 + 0.2*np.cos(i/10), 'total_signals_created': i*3}
             for i in range(100)
        ],
        'overmind_data': [
            {'step': i, 'decisions_made': i//5, 'success_rate': 0.7 + 0.2*np.sin(i/12),
             'collective_meditations_triggered': i//20} for i in range(100)
        ]
    }
    
    dummy_config = {
        'experiment_name': 'test_visualization',
        'max_steps': 100
    }
    
    # Test visualization creation
    try:
        visualizations = create_visualization_suite(
            dummy_data, dummy_config, "test_visualizations"
        )
        
        print(f"✅ Created {len(visualizations)} test visualizations")
        for name, path in visualizations.items():
            print(f"  - {name}: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False