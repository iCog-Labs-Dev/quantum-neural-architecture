import numpy as np
from itertools import product


def bars_and_stripes(grid_size=2):
    """
    Generate the Bars-and-Stripes (BAS) target distribution.

    A valid BAS pattern on an n x n grid is one where every row is
    all-0 or all-1 AND every column is all-0 or all-1.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid.

    Returns
    -------
    target : ndarray of shape (2^(grid_size^2),)
        Uniform probability over valid BAS patterns, 0 elsewhere.
    valid_indices : list of int
        Indices of the valid patterns in the probability vector.
    """
    n_bits = grid_size * grid_size
    n_states = 2 ** n_bits

    valid_indices = []
    for bits in product([0, 1], repeat=n_bits):
        grid = np.array(bits).reshape(grid_size, grid_size)
        rows_ok = all(
            len(set(grid[r, :])) == 1 for r in range(grid_size)
        )
        cols_ok = all(
            len(set(grid[:, c])) == 1 for c in range(grid_size)
        )
        if rows_ok and cols_ok:
            idx = int("".join(str(b) for b in bits), 2)
            valid_indices.append(idx)

    target = np.zeros(n_states)
    target[valid_indices] = 1.0 / len(valid_indices)
    return target, valid_indices


def bitstring_to_grid(bitstring, grid_size):
    """Reshape a flat bitstring (tuple/list/array) into an n x n grid."""
    return np.array(bitstring).reshape(grid_size, grid_size)


def index_to_bitstring(index, n_bits):
    """Convert an integer index to a binary tuple of length n_bits."""
    return tuple(int(b) for b in format(index, f"0{n_bits}b"))


def empirical_distribution(samples, n_bits):
    """
    Convert raw bitstring samples into a normalised probability vector.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n_bits)
    n_bits : int

    Returns
    -------
    ndarray of shape (2^n_bits,)
    """
    n_states = 2 ** n_bits
    counts = np.zeros(n_states)
    for row in samples:
        idx = int("".join(str(int(b)) for b in row), 2)
        counts[idx] += 1
    return counts / counts.sum()
