"""
Baseline Agents
===============
Communication and coordination baselines for comparison.

Implementations:
- CommNet: Learned continuous communication via averaging
- TarMAC: Targeted multi-agent communication with attention
- QMIX: Value decomposition with monotonic mixing
"""

from .commnet import CommNet, CommNetAgent, CommNetLayer
from .tarmac import TarMAC, TarMACAgent, TarMACLayer, MultiHeadAttention
from .qmix import QMIX, QMIXAgent, MixingNetwork, ReplayBuffer, PrioritizedReplayBuffer

__all__ = [
    # CommNet
    "CommNet",
    "CommNetAgent",
    "CommNetLayer",
    # TarMAC
    "TarMAC",
    "TarMACAgent",
    "TarMACLayer",
    "MultiHeadAttention",
    # QMIX
    "QMIX",
    "QMIXAgent",
    "MixingNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
]
