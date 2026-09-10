"""Bounded, reproducible evolution primitives independent of rendering and Torch."""
from .genome import Genome
from .mutation import mutate

__all__ = ['Genome', 'mutate']
