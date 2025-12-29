"""
MycoNet 3.0 UPRT Field Module
=============================

Unified Pattern Resonance Field implementation following field-theoretic principles.

Lagrangian formulation:
L_UPRT = -1/4 Tr(F^μν F_μν) + ψ̄(iγ^μ D_μ - m)ψ + V(Φ)

Where:
- F^μν is field strength tensor (curvature of gauge field Φ)
- ψ represents agent state contributions as spinor fields
- V(Φ) is potential encoding preferred field configurations
- Topological invariants (winding numbers Q) represent emergent symbols
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

from .config import UPRTFieldConfig

logger = logging.getLogger(__name__)


@dataclass
class TopologicalDefect:
    """Represents a topological defect in the field."""
    x: int
    y: int
    winding_number: float
    energy: float
    defect_type: str  # 'vortex', 'antivortex', 'monopole'
    creation_time: int


@dataclass
class FieldState:
    """Complete state of the UPRT field."""
    field_values: np.ndarray  # Complex field Φ(x,y)
    field_strength: np.ndarray  # F^μν tensor
    phase_map: np.ndarray  # Phase of field θ(x,y)
    energy_density: np.ndarray  # Local energy density
    total_energy: float
    topological_defects: List[TopologicalDefect]
    time_step: int


class UPRTField:
    """
    Unified Pattern Resonance Field representation.

    Implements gauge field on lattice with:
    - Field values Φ(x,y,t) (complex or vector-valued)
    - Curvature/field strength F^μν computation
    - Topological invariant calculation (winding numbers)
    - Energy functional E = ∫ L_UPRT d³x
    """

    def __init__(self, config):
        # Accept either MycoNetConfig or UPRTFieldConfig
        if hasattr(config, 'uprt_field'):
            self.config = config.uprt_field
        else:
            self.config = config
        self.resolution = self.config.field_resolution

        # Initialize complex field Φ(x,y)
        self.field = np.zeros((self.resolution, self.resolution), dtype=np.complex128)
        self._initialize_field()

        # Gauge field components A_μ
        self.gauge_field_x = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.gauge_field_y = np.zeros((self.resolution, self.resolution), dtype=np.float64)

        # Derived quantities
        self.field_strength = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.phase_map = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.energy_density = np.zeros((self.resolution, self.resolution), dtype=np.float64)

        # Topological tracking
        self.defects: List[TopologicalDefect] = []
        self.defect_history: List[List[TopologicalDefect]] = []

        # Time tracking
        self.time_step = 0

        # Lagrangian parameters
        self.g = self.config.coupling_constant  # Gauge coupling
        self.m = self.config.field_mass  # Field mass
        self.lambda_4 = self.config.potential_quartic  # Quartic coupling

        logger.info(f"UPRT Field initialized: {self.resolution}x{self.resolution} lattice")

    def _initialize_field(self):
        """Initialize field with random vacuum fluctuations."""
        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, (self.resolution, self.resolution))

        # Random amplitudes near vacuum expectation value
        amplitudes = np.abs(np.random.normal(1.0, 0.1, (self.resolution, self.resolution)))

        self.field = amplitudes * np.exp(1j * phases)

    def step_dynamics(self, agent_sources: Optional[Dict[int, Tuple[int, int, complex]]] = None,
                      dt: Optional[float] = None):
        """
        Evolve field using Euler-Lagrange equations.

        The equation of motion derived from L_UPRT is:
        ∂²Φ/∂t² = ∇²Φ - m²Φ - λ|Φ|²Φ + J_agent

        Simplified to first-order for numerical stability:
        ∂Φ/∂t = D∇²Φ - ∂V/∂Φ* + J_agent
        """
        if dt is None:
            dt = self.config.dt

        self.time_step += 1

        # Compute Laplacian ∇²Φ
        laplacian = self._compute_laplacian()

        # Compute potential gradient ∂V/∂Φ*
        potential_grad = self._compute_potential_gradient()

        # Add agent sources
        source_term = np.zeros_like(self.field)
        if agent_sources:
            for agent_id, (x, y, intensity) in agent_sources.items():
                if 0 <= x < self.resolution and 0 <= y < self.resolution:
                    source_term[y, x] += intensity

        # Euler-Lagrange evolution
        d_field = dt * (
                0.1 * laplacian  # Diffusion
                - potential_grad  # Potential gradient
                + source_term  # Agent sources
        )

        self.field += d_field

        # Clamp magnitude to prevent blow-up
        magnitude = np.abs(self.field)
        mask = magnitude > self.config.max_field_magnitude
        self.field[mask] *= self.config.max_field_magnitude / magnitude[mask]

        # Update derived quantities
        self._update_derived_quantities()

        # Detect topological defects
        self.defects = self._detect_topological_defects()
        self.defect_history.append(self.defects.copy())

    def _compute_laplacian(self) -> np.ndarray:
        """Compute Laplacian of field using finite differences."""
        # Periodic boundary conditions
        laplacian = (
                np.roll(self.field, 1, axis=0) +
                np.roll(self.field, -1, axis=0) +
                np.roll(self.field, 1, axis=1) +
                np.roll(self.field, -1, axis=1) -
                4 * self.field
        )
        return laplacian

    def _compute_potential_gradient(self) -> np.ndarray:
        """
        Compute gradient of potential V(Φ).

        V(Φ) = m²|Φ|² + λ|Φ|⁴

        ∂V/∂Φ* = m²Φ + 2λ|Φ|²Φ
        """
        magnitude_sq = np.abs(self.field) ** 2
        gradient = (
                self.m ** 2 * self.field +
                2 * self.lambda_4 * magnitude_sq * self.field
        )
        return gradient

    def _update_derived_quantities(self):
        """Update field strength, phase map, and energy density."""
        # Phase map
        self.phase_map = np.angle(self.field)

        # Field strength tensor F_xy (only spatial component in 2D)
        # F_xy = ∂_x A_y - ∂_y A_x
        self._update_gauge_field()
        self.field_strength = self._compute_field_strength()

        # Energy density
        self.energy_density = self._compute_energy_density()

    def _update_gauge_field(self):
        """Update gauge field components from phase gradient."""
        # A_x ≈ ∂_x θ (phase gradient approximation)
        self.gauge_field_x = np.diff(self.phase_map, axis=1, append=self.phase_map[:, :1])
        self.gauge_field_y = np.diff(self.phase_map, axis=0, append=self.phase_map[:1, :])

        # Handle phase wrapping
        self.gauge_field_x = np.mod(self.gauge_field_x + np.pi, 2 * np.pi) - np.pi
        self.gauge_field_y = np.mod(self.gauge_field_y + np.pi, 2 * np.pi) - np.pi

    def _compute_field_strength(self) -> np.ndarray:
        """Compute field strength tensor magnitude."""
        # F_xy = ∂_x A_y - ∂_y A_x
        d_y_Ax = np.diff(self.gauge_field_x, axis=0, append=self.gauge_field_x[:1, :])
        d_x_Ay = np.diff(self.gauge_field_y, axis=1, append=self.gauge_field_y[:, :1])

        F_xy = d_x_Ay - d_y_Ax

        # Handle phase wrapping
        F_xy = np.mod(F_xy + np.pi, 2 * np.pi) - np.pi

        return F_xy

    def _compute_energy_density(self) -> np.ndarray:
        """
        Compute local energy density from Lagrangian.

        E = 1/4 F_μν F^μν + |D_μ Φ|² + V(Φ)
        """
        # Kinetic term (simplified): |∇Φ|²
        grad_x = np.diff(self.field, axis=1, append=self.field[:, :1])
        grad_y = np.diff(self.field, axis=0, append=self.field[:1, :])
        kinetic = np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2

        # Potential term: m²|Φ|² + λ|Φ|⁴
        mag_sq = np.abs(self.field) ** 2
        potential = self.m ** 2 * mag_sq + self.lambda_4 * mag_sq ** 2

        # Field strength term: 1/4 F^2
        field_strength_term = 0.25 * self.field_strength ** 2

        return kinetic + potential + field_strength_term

    def detect_topological_defects(self) -> List[TopologicalDefect]:
        """Find vortices and monopoles in the field (public interface)."""
        return self._detect_topological_defects()

    def _detect_topological_defects(self) -> List[TopologicalDefect]:
        """
        Detect topological defects by computing winding numbers.

        A vortex exists where the phase winds by 2π around a plaquette.
        """
        defects = []

        # Compute winding number for each plaquette
        for y in range(self.resolution - 1):
            for x in range(self.resolution - 1):
                winding = self._compute_plaquette_winding(x, y)

                if abs(winding) > self.config.winding_number_threshold:
                    defect_type = 'vortex' if winding > 0 else 'antivortex'
                    defects.append(TopologicalDefect(
                        x=x, y=y,
                        winding_number=winding,
                        energy=float(self.energy_density[y, x]),
                        defect_type=defect_type,
                        creation_time=self.time_step
                    ))

        return defects

    def _compute_plaquette_winding(self, x: int, y: int) -> float:
        """Compute winding number around a plaquette."""
        # Get phases at corners
        phases = [
            self.phase_map[y, x],
            self.phase_map[y, (x + 1) % self.resolution],
            self.phase_map[(y + 1) % self.resolution, (x + 1) % self.resolution],
            self.phase_map[(y + 1) % self.resolution, x]
        ]

        # Compute phase differences along plaquette
        winding = 0.0
        for i in range(4):
            diff = phases[(i + 1) % 4] - phases[i]
            # Handle phase wrapping
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            winding += diff

        # Normalize to units of 2π
        return winding / (2 * np.pi)

    def compute_winding_number(self, region: Tuple[int, int, int, int]) -> float:
        """
        Compute total winding number for a rectangular region.

        Args:
            region: (x_min, y_min, x_max, y_max) defining the region
        """
        x_min, y_min, x_max, y_max = region
        total_winding = 0.0

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if 0 <= x < self.resolution - 1 and 0 <= y < self.resolution - 1:
                    total_winding += self._compute_plaquette_winding(x, y)

        return total_winding

    def get_field_energy(self) -> float:
        """Compute total field energy."""
        return float(np.sum(self.energy_density))

    def compute_phase_map(self) -> np.ndarray:
        """Extract phase field for coherence analysis."""
        return self.phase_map.copy()

    def add_agent_source(self, x: int, y: int, intensity: complex, spread: int = 2):
        """Add a local source from an agent."""
        for dy in range(-spread, spread + 1):
            for dx in range(-spread, spread + 1):
                nx = (x + dx) % self.resolution
                ny = (y + dy) % self.resolution
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance <= spread:
                    decay = np.exp(-distance / (spread / 2))
                    self.field[ny, nx] += intensity * decay

    def get_coherence_at(self, x: int, y: int, radius: int = 3) -> float:
        """Compute local phase coherence (Kuramoto-like order parameter)."""
        phases = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = (x + dx) % self.resolution
                ny = (y + dy) % self.resolution
                phases.append(self.phase_map[ny, nx])

        if not phases:
            return 0.0

        # Kuramoto order parameter: R = |1/N Σ e^(iθ_j)|
        complex_phases = np.exp(1j * np.array(phases))
        R = np.abs(np.mean(complex_phases))
        return float(R)

    def get_global_coherence(self) -> float:
        """Compute global phase coherence."""
        complex_phases = np.exp(1j * self.phase_map)
        R = np.abs(np.mean(complex_phases))
        return float(R)

    def get_state(self) -> FieldState:
        """Get complete field state."""
        return FieldState(
            field_values=self.field.copy(),
            field_strength=self.field_strength.copy(),
            phase_map=self.phase_map.copy(),
            energy_density=self.energy_density.copy(),
            total_energy=self.get_field_energy(),
            topological_defects=self.defects.copy(),
            time_step=self.time_step
        )

    def set_state(self, state: FieldState):
        """Set field state from FieldState object."""
        self.field = state.field_values.copy()
        self.time_step = state.time_step
        self._update_derived_quantities()
        self.defects = state.topological_defects.copy()

    def reset(self):
        """Reset field to initial state."""
        self.time_step = 0
        self._initialize_field()
        self.gauge_field_x.fill(0)
        self.gauge_field_y.fill(0)
        self.defects = []
        self.defect_history = []
        self._update_derived_quantities()

    def step(self, agent_states: np.ndarray = None, dt: float = None):
        """Alias for step_dynamics for compatibility."""
        # Convert agent_states array to agent_sources dict if provided
        agent_sources = None
        if agent_states is not None and len(agent_states) > 0:
            agent_sources = {}
            for i, state in enumerate(agent_states):
                if len(state) >= 5:
                    x = int(state[3] * self.resolution / 64) % self.resolution
                    y = int(state[4] * self.resolution / 64) % self.resolution
                    intensity = complex(state[0], state[1]) if len(state) > 1 else complex(1.0, 0.0)
                    agent_sources[i] = (x, y, intensity)
        self.step_dynamics(agent_sources, dt)

    def get_field_state(self) -> np.ndarray:
        """Get field values as numpy array."""
        return np.abs(self.field)

    def set_field_state(self, state: np.ndarray):
        """Set field values from numpy array."""
        if state.shape == self.field.shape:
            self.field = state.astype(np.complex128)
        elif len(state.shape) == 2:
            self.field = state.astype(np.complex128)
        self._update_derived_quantities()

    def detect_defects(self) -> List[TopologicalDefect]:
        """Alias for detect_topological_defects."""
        return self.detect_topological_defects()


if TORCH_AVAILABLE:
    class FieldSurrogateModel(nn.Module):
        """
        Neural network approximating UPRT field dynamics.

        Implements Fourier Neural Operator (FNO) architecture for learning
        PDE solutions efficiently.

        Input: (field_t, agent_positions, agent_states)
        Output: field_{t+1}
        """

        def __init__(self, resolution: int, hidden_dim: int = 64, num_layers: int = 4):
            super().__init__()
            self.resolution = resolution
            self.hidden_dim = hidden_dim

            # Input projection (2 channels: real and imaginary parts)
            self.input_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)

            # Fourier layers
            self.fourier_layers = nn.ModuleList([
                SpectralConv2d(hidden_dim, hidden_dim, resolution // 2)
                for _ in range(num_layers)
            ])

            # Non-linear layers
            self.conv_layers = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
                for _ in range(num_layers)
            ])

            # Layer normalization
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(hidden_dim)
                for _ in range(num_layers)
            ])

            # Output projection (back to 2 channels)
            self.output_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 2, kernel_size=1)
            )

        def forward(self, field_t: torch.Tensor,
                    agent_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass to predict next field state.

            Args:
                field_t: Current field state [batch, 2, H, W] (real, imag)
                agent_positions: Optional agent position encoding [batch, N, 2]

            Returns:
                Predicted field state [batch, 2, H, W]
            """
            # Ensure correct shape
            if field_t.dim() == 3:
                field_t = field_t.unsqueeze(0)

            # Input projection
            x = self.input_proj(field_t)

            # Fourier layers with residual connections
            for fourier, conv, norm in zip(self.fourier_layers, self.conv_layers, self.norms):
                # Spectral convolution
                x1 = fourier(x)

                # Local convolution
                x2 = conv(x)

                # Combine and normalize
                x = norm(x + x1 + x2)
                x = torch.relu(x)

            # Output projection
            output = self.output_proj(x)

            # Add residual connection from input
            output = output + field_t

            return output

        def predict_trajectory(self, field_t: torch.Tensor,
                               num_steps: int) -> List[torch.Tensor]:
            """Predict field trajectory for multiple time steps."""
            trajectory = [field_t]
            current = field_t

            for _ in range(num_steps):
                current = self.forward(current)
                trajectory.append(current)

            return trajectory


    class SpectralConv2d(nn.Module):
        """
        2D Spectral convolution layer using FFT.

        Implements the core operation of Fourier Neural Operator:
        - Transform to Fourier space
        - Apply learnable spectral weights
        - Transform back to physical space
        """

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes

            scale = 1 / (in_channels * out_channels)

            # Complex weights for Fourier modes
            self.weights1 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            )
            self.weights2 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]

            # FFT
            x_ft = torch.fft.rfft2(x)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batch_size, self.out_channels, x.shape[-2],
                                 x.shape[-1] // 2 + 1, dtype=torch.cfloat, device=x.device)

            # Upper half of Fourier modes
            out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :self.modes, :self.modes],
                self.weights1
            )

            # Lower half of Fourier modes
            out_ft[:, :, -self.modes:, :self.modes] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -self.modes:, :self.modes],
                self.weights2
            )

            # Inverse FFT
            x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))

            return x


    class UNetSurrogate(nn.Module):
        """
        U-Net style surrogate for UPRT field dynamics.

        Alternative to FNO for cases where local features are important.
        """

        def __init__(self, resolution: int, hidden_dim: int = 32):
            super().__init__()

            # Encoder
            self.enc1 = self._conv_block(2, hidden_dim)
            self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
            self.enc3 = self._conv_block(hidden_dim * 2, hidden_dim * 4)

            # Bottleneck
            self.bottleneck = self._conv_block(hidden_dim * 4, hidden_dim * 8)

            # Decoder
            self.dec3 = self._conv_block(hidden_dim * 8 + hidden_dim * 4, hidden_dim * 4)
            self.dec2 = self._conv_block(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2)
            self.dec1 = self._conv_block(hidden_dim * 2 + hidden_dim, hidden_dim)

            # Output
            self.output = nn.Conv2d(hidden_dim, 2, kernel_size=1)

            # Pooling and upsampling
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))

            # Bottleneck
            b = self.bottleneck(self.pool(e3))

            # Decoder with skip connections
            d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

            # Output with residual connection
            return self.output(d1) + x

else:
    # Fallback classes when PyTorch is not available
    class FieldSurrogateModel:
        """Placeholder for FieldSurrogateModel when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not available. Using dummy FieldSurrogateModel.")

        def forward(self, *args, **kwargs):
            return None


    class SpectralConv2d:
        """Placeholder for SpectralConv2d when PyTorch is unavailable."""
        pass


    class UNetSurrogate:
        """Placeholder for UNetSurrogate when PyTorch is unavailable."""
        pass


def create_field_surrogate(architecture: str, resolution: int, hidden_dim: int = 64):
    """
    Factory function to create field surrogate models.

    Args:
        architecture: 'fno', 'unet', or 'gnn'
        resolution: Field grid resolution
        hidden_dim: Hidden dimension size
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Cannot create field surrogate.")
        return None

    if architecture.lower() == 'fno':
        return FieldSurrogateModel(resolution, hidden_dim)
    elif architecture.lower() == 'unet':
        return UNetSurrogate(resolution, hidden_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
