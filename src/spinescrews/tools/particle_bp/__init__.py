"""Particle belief propagation (D-PMP) for tree-structured optimization.

Implements discrete-continuous particle message passing on factor graphs,
following Zuffi & Black, CVPR 2015.
"""

from .graph import FactorGraph
from .particles import ParticleSet
from .proposals import RandomWalkProposal, NeighborProposal
from .solver import ParticleBPSolver, SolverConfig, SolverResult

__all__ = [
    'FactorGraph',
    'ParticleSet',
    'RandomWalkProposal',
    'NeighborProposal',
    'ParticleBPSolver',
    'SolverConfig',
    'SolverResult',
]
