"""
MycoNet 3.0 Hypernetwork Module
===============================

QREA v3.0 Hypernetwork Compression: H_θ: z^(agent) → W^(agent)

Maps low-dimensional genome vectors to full agent network weights,
enabling evolutionary optimization in compressed parameter space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .config import QREAConfig

logger = logging.getLogger(__name__)


@dataclass
class LayerSpec:
    """Specification for a layer in the target architecture."""
    name: str
    input_dim: int
    output_dim: int
    has_bias: bool = True


@dataclass
class TargetArchitecture:
    """Specification of the target agent network architecture."""
    layers: List[LayerSpec]

    @property
    def total_params(self) -> int:
        """Total number of parameters in target network."""
        total = 0
        for layer in self.layers:
            total += layer.input_dim * layer.output_dim
            if layer.has_bias:
                total += layer.output_dim
        return total

    @classmethod
    def default_agent_architecture(cls) -> 'TargetArchitecture':
        """Create default agent network architecture."""
        return cls(layers=[
            LayerSpec('policy.encoder.0', 64, 128),
            LayerSpec('policy.encoder.2', 128, 128),
            LayerSpec('policy.action_head', 128, 13),  # 13 action types
            LayerSpec('world_model.encoder', 64, 128),
            LayerSpec('world_model.predictor', 128, 64),
            LayerSpec('value.fc1', 64, 64),
            LayerSpec('value.fc2', 64, 1),
        ])


if TORCH_AVAILABLE:
    class GenomeHyperNet(nn.Module):
        """
        Maps low-dimensional genome vectors to full agent network weights.

        Architecture:
        - Input: genome vector z (dim ~200-500)
        - Output: Weight matrices for policy_net, world_model, etc.
        - Uses weight generation via layer-by-layer MLPs
        """

        def __init__(self, genome_dim: int, target_architecture: TargetArchitecture,
                     hidden_dim: int = 512):
            super().__init__()
            self.genome_dim = genome_dim
            self.target_architecture = target_architecture
            self.hidden_dim = hidden_dim

            # Create weight generators for each layer
            self.weight_generators = nn.ModuleDict()
            self.bias_generators = nn.ModuleDict()

            for layer_spec in target_architecture.layers:
                weight_size = layer_spec.input_dim * layer_spec.output_dim

                # Weight generator network
                self.weight_generators[layer_spec.name] = nn.Sequential(
                    nn.Linear(genome_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, weight_size),
                )

                # Bias generator (if needed)
                if layer_spec.has_bias:
                    self.bias_generators[layer_spec.name] = nn.Sequential(
                        nn.Linear(genome_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, layer_spec.output_dim),
                    )

            # Global conditioning network
            self.global_conditioning = nn.Sequential(
                nn.Linear(genome_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, genome_dim),
            )

            # Initialize with proper scaling
            self._initialize_weights()

            logger.info(f"HyperNetwork initialized: genome_dim={genome_dim}, "
                        f"total_params={target_architecture.total_params}")

        def _initialize_weights(self):
            """Initialize hypernetwork weights for stable generation."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Smaller initialization for output layers
                    if module.out_features > 1000:
                        nn.init.normal_(module.weight, std=0.01)
                    else:
                        nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, genome_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Generate full agent network parameters from genome.

            Args:
                genome_vector: Tensor of shape (batch_size, genome_dim) or (genome_dim,)

            Returns:
                Dict of weight tensors: {'layer_name.weight': tensor, 'layer_name.bias': tensor}
            """
            # Handle single genome
            if genome_vector.dim() == 1:
                genome_vector = genome_vector.unsqueeze(0)

            batch_size = genome_vector.size(0)

            # Apply global conditioning
            conditioned_genome = genome_vector + self.global_conditioning(genome_vector)

            # Generate weights for each layer
            weights = {}

            for layer_spec in self.target_architecture.layers:
                # Generate weights
                weight_flat = self.weight_generators[layer_spec.name](conditioned_genome)
                weight_matrix = weight_flat.view(
                    batch_size, layer_spec.output_dim, layer_spec.input_dim
                )

                # Scale weights for proper initialization
                weight_matrix = weight_matrix * np.sqrt(2.0 / layer_spec.input_dim)

                weights[f'{layer_spec.name}.weight'] = weight_matrix

                # Generate biases
                if layer_spec.has_bias:
                    bias = self.bias_generators[layer_spec.name](conditioned_genome)
                    weights[f'{layer_spec.name}.bias'] = bias

            return weights

        def generate_agent_weights(self, genome_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Convenience method for single agent weight generation."""
            weights = self.forward(genome_vector)

            # Remove batch dimension for single agent
            if genome_vector.dim() == 1:
                weights = {k: v.squeeze(0) for k, v in weights.items()}

            return weights

        def genome_to_agent_state_dict(self, genome_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Convert genome to a state dict that can be loaded into an agent network.
            """
            return self.generate_agent_weights(genome_vector)


    class EvolutionEngine:
        """
        Evolutionary algorithm for optimizing genomes.

        Implements:
        - Selection (tournament, rank-based, or fitness-proportional)
        - Crossover (uniform, single-point, or blend)
        - Mutation (Gaussian noise, polynomial mutation)
        - Elitism (preserve top K individuals)
        """

        def __init__(self, config: QREAConfig, genome_dim: int):
            self.config = config
            self.genome_dim = genome_dim
            self.population_size = config.population_size
            self.mutation_rate = config.mutation_rate
            self.mutation_strength = config.mutation_strength
            self.crossover_rate = config.crossover_rate
            self.elite_fraction = config.elite_fraction
            self.tournament_size = config.tournament_size

            # Population
            self.population: Optional[torch.Tensor] = None
            self.fitness_history: List[List[float]] = []
            self.generation = 0

            # Best genome tracking
            self.best_genome: Optional[torch.Tensor] = None
            self.best_fitness: float = float('-inf')

        def initialize_population(self) -> torch.Tensor:
            """
            Initialize random population of genome vectors.
            """
            # Initialize with normalized random vectors
            self.population = torch.randn(self.population_size, self.genome_dim)
            self.population = self.population / self.population.norm(dim=1, keepdim=True)

            logger.info(f"Initialized population: {self.population_size} genomes, "
                        f"dimension {self.genome_dim}")

            return self.population

        def evolve_population(self, genomes: torch.Tensor,
                              fitness_scores: torch.Tensor) -> torch.Tensor:
            """
            Run one generation of evolution.

            Args:
                genomes: Population tensor [population_size, genome_dim]
                fitness_scores: Fitness for each genome [population_size]

            Returns:
                New population of genome vectors
            """
            self.generation += 1
            population_size = genomes.size(0)

            # Convert to numpy for easier manipulation
            genomes_np = genomes.detach().cpu().numpy()
            fitness_np = fitness_scores.detach().cpu().numpy()

            # Track fitness history
            self.fitness_history.append(fitness_np.tolist())

            # Update best genome
            best_idx = np.argmax(fitness_np)
            if fitness_np[best_idx] > self.best_fitness:
                self.best_fitness = fitness_np[best_idx]
                self.best_genome = genomes[best_idx].clone()

            # 1. Elitism - preserve top individuals
            elite_count = max(1, int(population_size * self.elite_fraction))
            elite_indices = np.argsort(fitness_np)[-elite_count:]
            elites = genomes_np[elite_indices].copy()

            # 2. Selection - tournament selection for parents
            parents = self._tournament_selection(genomes_np, fitness_np, population_size)

            # 3. Crossover
            offspring = self._crossover(parents)

            # 4. Mutation
            offspring = self._mutate(offspring)

            # 5. Combine elites and offspring
            num_offspring = population_size - elite_count
            new_population = np.vstack([elites, offspring[:num_offspring]])

            # Convert back to tensor
            new_genomes = torch.from_numpy(new_population).float()
            self.population = new_genomes

            logger.debug(f"Generation {self.generation}: best_fitness={self.best_fitness:.4f}, "
                         f"mean_fitness={np.mean(fitness_np):.4f}")

            return new_genomes

        def _tournament_selection(self, genomes: np.ndarray, fitness: np.ndarray,
                                  num_parents: int) -> np.ndarray:
            """Tournament selection for parent genomes."""
            parents = []

            for _ in range(num_parents):
                # Select random individuals for tournament
                tournament_indices = np.random.choice(
                    len(genomes), size=self.tournament_size, replace=False
                )
                tournament_fitness = fitness[tournament_indices]

                # Winner is the one with highest fitness
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parents.append(genomes[winner_idx])

            return np.array(parents)

        def _crossover(self, parents: np.ndarray) -> np.ndarray:
            """Apply crossover to parent pairs."""
            offspring = []
            num_parents = len(parents)

            for i in range(0, num_parents - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < num_parents else parents[0]

                if np.random.random() < self.crossover_rate:
                    # Blend crossover (BLX-α)
                    alpha = 0.5
                    beta = np.random.uniform(-alpha, 1 + alpha, size=self.genome_dim)
                    child1 = beta * parent1 + (1 - beta) * parent2
                    child2 = (1 - beta) * parent1 + beta * parent2
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()

                offspring.append(child1)
                offspring.append(child2)

            # Handle odd number of parents
            if num_parents % 2 == 1:
                offspring.append(parents[-1].copy())

            return np.array(offspring)

        def _mutate(self, offspring: np.ndarray) -> np.ndarray:
            """Apply Gaussian mutation to offspring."""
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_rate:
                    # Gaussian mutation
                    mutation = np.random.normal(0, self.mutation_strength, size=self.genome_dim)
                    offspring[i] = offspring[i] + mutation

                    # Occasionally re-normalize
                    if np.random.random() < 0.1:
                        norm = np.linalg.norm(offspring[i])
                        if norm > 0:
                            offspring[i] = offspring[i] / norm

            return offspring

        def get_best_genome(self) -> Optional[torch.Tensor]:
            """Get the best genome found so far."""
            return self.best_genome

        def get_population_stats(self) -> Dict[str, float]:
            """Get statistics about current population."""
            if not self.fitness_history:
                return {}

            latest_fitness = np.array(self.fitness_history[-1])

            return {
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': float(np.mean(latest_fitness)),
                'std_fitness': float(np.std(latest_fitness)),
                'min_fitness': float(np.min(latest_fitness)),
                'population_size': self.population_size,
                'genome_dim': self.genome_dim
            }

        def save_state(self, filepath: str):
            """Save evolution state."""
            state = {
                'population': self.population,
                'best_genome': self.best_genome,
                'best_fitness': self.best_fitness,
                'fitness_history': self.fitness_history,
                'generation': self.generation,
                'config': {
                    'population_size': self.population_size,
                    'genome_dim': self.genome_dim,
                    'mutation_rate': self.mutation_rate,
                    'mutation_strength': self.mutation_strength,
                    'crossover_rate': self.crossover_rate,
                    'elite_fraction': self.elite_fraction,
                }
            }
            torch.save(state, filepath)
            logger.info(f"Evolution state saved to {filepath}")

        def load_state(self, filepath: str):
            """Load evolution state."""
            state = torch.load(filepath)
            self.population = state['population']
            self.best_genome = state['best_genome']
            self.best_fitness = state['best_fitness']
            self.fitness_history = state['fitness_history']
            self.generation = state['generation']
            logger.info(f"Evolution state loaded from {filepath}")


    class HierarchicalMERA:
        """
        Multi-scale entanglement renormalization ansatz for abstraction.

        Implements hierarchical structure for learning at multiple scales:
        - Fine-grained local features at bottom
        - Increasingly abstract representations at higher levels
        """

        def __init__(self, num_sites: int, num_levels: int = 4, bond_dim: int = 16):
            self.num_sites = num_sites
            self.num_levels = num_levels
            self.bond_dim = bond_dim

            # Isometry tensors (coarse-graining)
            self.isometries = nn.ModuleList([
                nn.Linear(2 * bond_dim, bond_dim)
                for _ in range(num_levels)
            ])

            # Disentangler tensors
            self.disentanglers = nn.ModuleList([
                nn.Linear(2 * bond_dim, 2 * bond_dim)
                for _ in range(num_levels)
            ])

            # Top tensor
            self.top_tensor = nn.Parameter(torch.randn(bond_dim))

        def forward(self, site_tensors: torch.Tensor) -> torch.Tensor:
            """
            Compute hierarchical representation.

            Args:
                site_tensors: Tensor of shape [batch, num_sites, bond_dim]

            Returns:
                Top-level representation [batch, bond_dim]
            """
            current = site_tensors

            for level in range(self.num_levels):
                # Apply disentanglers to pairs
                batch_size, num_sites, dim = current.shape

                if num_sites < 2:
                    break

                # Pair sites
                pairs = current.view(batch_size, num_sites // 2, 2 * dim)

                # Disentangle
                disentangled = self.disentanglers[level](pairs)
                disentangled = F.relu(disentangled)

                # Coarse-grain with isometries
                coarse = self.isometries[level](disentangled)
                current = coarse

            # Return mean of remaining sites
            return current.mean(dim=1)


else:
    # Fallback classes when PyTorch is not available
    class GenomeHyperNet:
        """Placeholder for GenomeHyperNet when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not available. Using dummy GenomeHyperNet.")
            self.genome_dim = kwargs.get('genome_dim', 256)

        def forward(self, genome_vector):
            return {}


    class EvolutionEngine:
        """Placeholder for EvolutionEngine when PyTorch is unavailable."""

        def __init__(self, config, genome_dim):
            self.config = config
            self.genome_dim = genome_dim
            self.population = None
            self.generation = 0
            logger.warning("PyTorch not available. Using NumPy-based EvolutionEngine.")

        def initialize_population(self):
            population_size = self.config.population_size
            self.population = np.random.randn(population_size, self.genome_dim)
            return self.population

        def evolve_population(self, genomes, fitness_scores):
            # Simple evolutionary update
            fitness = np.array(fitness_scores)
            elite_count = max(1, int(len(fitness) * 0.1))
            elite_indices = np.argsort(fitness)[-elite_count:]

            new_pop = genomes[elite_indices].copy()

            while len(new_pop) < len(genomes):
                parent_idx = np.random.choice(elite_indices)
                child = genomes[parent_idx] + np.random.randn(self.genome_dim) * 0.1
                new_pop = np.vstack([new_pop, child])

            self.generation += 1
            return new_pop[:len(genomes)]


    class HierarchicalMERA:
        """Placeholder for HierarchicalMERA when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not available. Using dummy HierarchicalMERA.")


def create_hypernetwork(genome_dim: int = 256,
                        architecture: Optional[TargetArchitecture] = None,
                        hidden_dim: int = 512) -> GenomeHyperNet:
    """
    Factory function to create a hypernetwork.

    Args:
        genome_dim: Dimension of genome vectors
        architecture: Target network architecture specification
        hidden_dim: Hidden dimension in hypernetwork MLPs
    """
    if architecture is None:
        architecture = TargetArchitecture.default_agent_architecture()

    return GenomeHyperNet(genome_dim, architecture, hidden_dim)


def create_evolution_engine(config: QREAConfig,
                            genome_dim: int = 256) -> EvolutionEngine:
    """
    Factory function to create an evolution engine.

    Args:
        config: QREA configuration
        genome_dim: Dimension of genome vectors
    """
    return EvolutionEngine(config, genome_dim)
