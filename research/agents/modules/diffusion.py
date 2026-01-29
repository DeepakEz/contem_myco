"""
Module B: Stigmergic Diffusion Field
=====================================
Mycelial-inspired signal propagation for decentralized coordination.

Key innovation: Agents communicate through a shared spatial field rather than
direct message passing, enabling scalable coordination with local computation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SignalChannel:
    """Definition of a signal channel."""
    name: str
    decay_rate: float = 0.05
    diffusion_rate: float = 0.1


# Default signal channels
DEFAULT_CHANNELS = [
    SignalChannel("danger", decay_rate=0.1, diffusion_rate=0.15),
    SignalChannel("resource", decay_rate=0.05, diffusion_rate=0.1),
    SignalChannel("coordination", decay_rate=0.03, diffusion_rate=0.2),
    SignalChannel("ethics", decay_rate=0.02, diffusion_rate=0.05),
]


class DiffusionField:
    """
    Spatial diffusion field for stigmergic communication.

    Agents deposit signals that diffuse and decay over time.
    Other agents sense local gradients to coordinate behavior.
    """

    def __init__(
        self,
        grid_size: int = 32,
        num_channels: int = 4,
        world_bounds: Tuple[float, float] = (-1.0, 1.0),
        channels: Optional[List[SignalChannel]] = None,
    ):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.world_bounds = world_bounds
        self.channels = channels or DEFAULT_CHANNELS[:num_channels]

        # Initialize field: (channels, height, width)
        self.field = np.zeros((num_channels, grid_size, grid_size), dtype=np.float32)

        # Precompute diffusion kernels
        self._diffusion_kernel = self._create_diffusion_kernel()

    def _create_diffusion_kernel(self) -> np.ndarray:
        """Create normalized diffusion kernel (Gaussian-like)."""
        kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ], dtype=np.float32)
        return kernel / kernel.sum()

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        # Normalize to [0, 1]
        normalized = (pos - self.world_bounds[0]) / (self.world_bounds[1] - self.world_bounds[0])
        # Scale to grid
        grid_pos = (normalized * (self.grid_size - 1)).astype(int)
        # Clamp to bounds
        grid_pos = np.clip(grid_pos, 0, self.grid_size - 1)
        return tuple(grid_pos)

    def deposit(
        self,
        position: np.ndarray,
        channel: int,
        strength: float = 1.0,
        radius: int = 1
    ):
        """
        Deposit signal at a position.

        Args:
            position: World coordinates (x, y)
            channel: Channel index
            strength: Signal strength
            radius: Deposit radius in grid cells
        """
        gx, gy = self.world_to_grid(position)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    falloff = np.exp(-dist / max(radius, 1))
                    self.field[channel, ny, nx] += strength * falloff

    def sense(
        self,
        position: np.ndarray,
        radius: int = 3
    ) -> np.ndarray:
        """
        Sense local field values and gradients.

        Args:
            position: World coordinates
            radius: Sensing radius in grid cells

        Returns:
            Feature vector: [channel_values (C), gradient_x (C), gradient_y (C)]
            Shape: (3 * num_channels,)
        """
        gx, gy = self.world_to_grid(position)

        features = []

        for c in range(self.num_channels):
            # Local value (average in radius)
            values = []
            grad_x, grad_y = 0.0, 0.0

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        val = self.field[c, ny, nx]
                        values.append(val)
                        # Gradient contribution
                        grad_x += dx * val
                        grad_y += dy * val

            avg_value = np.mean(values) if values else 0.0
            norm = len(values) if values else 1

            features.extend([
                avg_value,
                grad_x / norm,
                grad_y / norm
            ])

        return np.array(features, dtype=np.float32)

    def step(self):
        """
        Advance field by one timestep: diffuse and decay.
        """
        new_field = np.zeros_like(self.field)

        for c in range(self.num_channels):
            channel = self.channels[c] if c < len(self.channels) else self.channels[0]

            # Diffusion via convolution
            from scipy.ndimage import convolve
            diffused = convolve(self.field[c], self._diffusion_kernel, mode='constant')

            # Blend original with diffused
            new_field[c] = (1 - channel.diffusion_rate) * self.field[c] + \
                           channel.diffusion_rate * diffused

            # Decay
            new_field[c] *= (1 - channel.decay_rate)

        self.field = new_field

    def reset(self):
        """Reset field to zero."""
        self.field = np.zeros_like(self.field)

    def get_visualization(self) -> np.ndarray:
        """Get field state for visualization."""
        return self.field.copy()


class DiffusionEncoder(nn.Module):
    """
    Neural network to encode diffusion field observations.
    """

    def __init__(
        self,
        num_channels: int = 4,
        sense_radius: int = 3,
        hidden_size: int = 64,
        output_size: int = 32
    ):
        super().__init__()

        # Input: 3 features per channel (value, grad_x, grad_y)
        input_size = 3 * num_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, field_obs: torch.Tensor) -> torch.Tensor:
        """
        Encode field observation.

        Args:
            field_obs: (batch, 3 * num_channels)

        Returns:
            encoded: (batch, output_size)
        """
        return self.encoder(field_obs)


class DiffusionPolicy(nn.Module):
    """
    Policy head for determining what signals to deposit.
    """

    def __init__(
        self,
        state_size: int,
        num_channels: int = 4,
        hidden_size: int = 64
    ):
        super().__init__()

        self.num_channels = num_channels

        self.deposit_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_channels),
            nn.Sigmoid()  # Deposit strengths in [0, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict deposit strengths for each channel.

        Args:
            state: Agent's internal state (batch, state_size)

        Returns:
            deposits: (batch, num_channels) in [0, 1]
        """
        return self.deposit_net(state)


# Convenience functions for integration
def create_diffusion_module(config) -> Tuple[DiffusionField, DiffusionEncoder, DiffusionPolicy]:
    """Create all diffusion components from config."""
    field = DiffusionField(
        grid_size=config.grid_size,
        num_channels=config.num_channels,
    )

    encoder = DiffusionEncoder(
        num_channels=config.num_channels,
        sense_radius=config.sense_radius,
    )

    policy = DiffusionPolicy(
        state_size=256,  # Will be set properly during integration
        num_channels=config.num_channels,
    )

    return field, encoder, policy
