"""Record Linkage Deduplicator.

Probabilistic record linkage (Fellegi-Sunter model) with Jaro-Winkler
fuzzy string comparison, blocking, and an optional EM step to estimate
m/u agreement probabilities from the data itself.
"""

from .similarity import jaro, jaro_winkler
from .model import FieldSpec, FellegiSunterModel
from .blocking import build_blocking_keys
from .linker import run_linkage

__all__ = [
    "jaro",
    "jaro_winkler",
    "FieldSpec",
    "FellegiSunterModel",
    "build_blocking_keys",
    "run_linkage",
]
