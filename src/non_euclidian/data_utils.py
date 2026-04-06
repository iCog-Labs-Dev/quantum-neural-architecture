"""
Data utilities for non-Euclidean VQC experiments.

Provides coordinate transforms that preserve geometry so the quantum
embedding can exploit the Bloch-sphere inductive bias.
"""

import numpy as np


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



def make_sphere_moons(n_samples=300, noise_std=0.08, seed=42):
    """
    Two crescent-shaped classes on the surface of a unit sphere.
    Both theta (latitude) and phi (longitude) determine the class.
    Radius is constant -- geometry is purely spherical.

    Class 0: centred at latitude 60 deg, eastern hemisphere (phi in [0, pi]).
    Class 1: centred at latitude 120 deg, western hemisphere (phi in [pi, 2*pi]).

    Parameters
    ----------
    n_samples : int
    noise_std : float
        Controls Gaussian spread of the latitude band (scaled by pi).
    seed : int

    Returns
    -------
    theta  : ndarray (n_samples,)   -- polar angle in [0, pi].
    phi    : ndarray (n_samples,)   -- azimuthal angle in [0, 2*pi).
    labels : ndarray (n_samples,)   -- 0 or 1.
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    theta_0 = rng.normal(np.pi / 3, noise_std * np.pi, n_half)
    phi_0 = rng.uniform(0, np.pi, n_half)

    theta_1 = rng.normal(2 * np.pi / 3, noise_std * np.pi, n_half)
    phi_1 = rng.uniform(np.pi, 2 * np.pi, n_half)

    theta = np.concatenate([theta_0, theta_1])
    phi = np.concatenate([phi_0, phi_1])
    labels = np.array([0.0] * n_half + [1.0] * n_half)

    theta = np.clip(theta, 0, np.pi)
    phi = phi % (2 * np.pi)

    return theta, phi, labels


def sphere_to_cartesian(theta, phi):
    """Convert spherical (theta, phi) to Cartesian (x, y, z) on the unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])
