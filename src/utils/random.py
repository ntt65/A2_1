"""Explicit experiment-entry-point seeding without import-time RNG changes."""

import numpy as np

__concepts__ = {"Reproducible_Seed": ["set_seed"]}


def set_seed(seed: int = 42) -> None:
    """Seed NumPy's legacy global RNG once at an experiment's entry point.

    seed is an integer scalar in [0, 2**32 - 1]; no tensor is returned.
    Booleans and non-integers raise TypeError; out-of-range values raise
    ValueError. Importing this module never changes the RNG state.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must lie between 0 and 2**32 - 1")
    np.random.seed(int(seed))
