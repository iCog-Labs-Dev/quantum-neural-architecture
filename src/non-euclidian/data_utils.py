"""
Data utilities for non-Euclidean VQC experiments.

Provides coordinate transforms that preserve geometry so the quantum
embedding can exploit the Bloch-sphere inductive bias.
"""

import numpy as np


# ------------------------------------------------------------------
# Cartesian  -->  Polar   (for make_circles / 2-D datasets)
# ------------------------------------------------------------------

def to_polar(X):
    """
    Convert Cartesian (x, y) to polar (r, theta).

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Cartesian coordinates.

    Returns
    -------
    ndarray of shape (n_samples, 2)
        Column 0 = r  in [0, inf),  column 1 = theta in [0, 2*pi).
    """
    x, y = X[:, 0], X[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    return np.column_stack([r, theta])


def polar_to_bloch(X_polar):
    """
    Prepare polar features for the *spherical* embedding.

    The spherical embedding expects pairs [theta_0, phi_0, ...] where
    theta controls RY (polar angle on Bloch sphere, range [0, pi]) and
    phi controls RZ (azimuthal angle, range [0, 2*pi]).

    We scale *r* into [0, pi] and keep *theta* in [0, 2*pi].
    """
    r = X_polar[:, 0]
    theta = X_polar[:, 1]

    r_min, r_max = r.min(), r.max()
    if r_max - r_min > 1e-12:
        r_scaled = (r - r_min) / (r_max - r_min) * np.pi
    else:
        r_scaled = np.full_like(r, np.pi / 2)

    return np.column_stack([r_scaled, theta])


# ------------------------------------------------------------------
# Cyclical feature encoding
# ------------------------------------------------------------------

def to_cyclical(values, period):
    """
    Convert a scalar periodic feature to sin/cos pair (for classical models)
    and to a raw angle (for the VQC cyclical embedding).

    Parameters
    ----------
    values : ndarray of shape (n,)
        Raw periodic values (e.g. hour 0-23).
    period : float
        Full period length (e.g. 24 for hours).

    Returns
    -------
    angles   : ndarray (n,)   -- 2*pi*values/period, for VQC RZ embedding.
    sin_cos  : ndarray (n, 2) -- [sin(angle), cos(angle)], for classical FFNN.
    """
    angles = 2 * np.pi * values / period
    sin_cos = np.column_stack([np.sin(angles), np.cos(angles)])
    return angles, sin_cos


# ------------------------------------------------------------------
# Synthetic cyclical dataset
# ------------------------------------------------------------------

def generate_cyclical_dataset(n_samples=300, noise_std=0.5, seed=42):
    """
    Create a binary classification task where the decision boundary
    wraps around midnight.

    Positive class:  hours roughly in [21, 5)  (late night / early morning).
    Negative class:  hours roughly in [5, 21).

    The wrap-around at 0/24 is what makes this hard for a vanilla FFNN
    that sees hour as a plain scalar.

    Parameters
    ----------
    n_samples : int
    noise_std : float
        Gaussian noise added to the hour value.
    seed : int

    Returns
    -------
    hours  : ndarray (n_samples,)   -- noisy hour values in [0, 24).
    labels : ndarray (n_samples,)   -- 0 or 1.
    """
    rng = np.random.default_rng(seed)
    hours_clean = rng.uniform(0, 24, n_samples)
    hours = (hours_clean + rng.normal(0, noise_std, n_samples)) % 24

    labels = np.where(
        (hours_clean >= 21) | (hours_clean < 5), 1.0, 0.0
    )
    return hours, labels


# ------------------------------------------------------------------
# Spherical dataset (3-D Cartesian on unit sphere)
# ------------------------------------------------------------------

def cartesian_to_spherical(X):
    """
    Convert 3D Cartesian (x, y, z) to spherical (theta, phi).
    r is assumed to be 1 for points on the unit sphere.

    theta: polar angle in [0, pi] (angle from z-axis).
    phi: azimuthal angle in [0, 2*pi] (angle in x-y plane).
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Clip to avoid numerical errors with arccos
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2 * np.pi)
    return np.column_stack([theta, phi])


def generate_spherical_dataset(n_samples=400, noise=0.1, seed=42):
    """
    Generate points on a 3D unit sphere.
    Class 1: northern hemisphere (z > 0).
    Class 0: southern hemisphere (z < 0).
    """
    rng = np.random.default_rng(seed)

    # Uniform sampling on a sphere
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    cos_theta = rng.uniform(-1, 1, n_samples)
    theta = np.arccos(cos_theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    X = np.column_stack([x, y, z])
    # Add noise
    X += rng.normal(0, noise, X.shape)
    # Re-normalize to unit sphere (optional, but keeps it "spherical")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= norms

    # Label based on the original z position (before noise/normalization)
    labels = np.where(cos_theta > 0, 1.0, 0.0)

    return X, labels
