"""Interchangeable policies. Neural dependencies are imported only by adapters."""
from .baseline import RandomAgent, HeuristicAgent

__all__ = ['RandomAgent', 'HeuristicAgent']
