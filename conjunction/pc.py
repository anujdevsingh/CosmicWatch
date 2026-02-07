from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PcResult:
    pc: float
    method: str
    notes: str = ""


def _as_vec2(mu: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    if mu_arr.shape != (2,):
        raise ValueError("mu must be length-2")
    return mu_arr


def _as_cov2(cov: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.shape != (2, 2):
        raise ValueError("cov must be 2x2")
    cov_arr = (cov_arr + cov_arr.T) / 2.0
    return cov_arr


def _bvnorm_pdf(xy: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    det = float(np.linalg.det(cov))
    if det <= 0.0:
        raise ValueError("cov must be positive definite")
    d = xy - mu
    quad = np.einsum("...i,ij,...j->...", d, inv, d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    return norm * np.exp(-0.5 * quad)


def pc_circle_grid(
    mu: np.ndarray | list[float] | tuple[float, float],
    cov: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
    radius: float,
    *,
    n_theta: int = 720,
    n_r: int = 400,
) -> PcResult:
    mu2 = _as_vec2(mu)
    cov2 = _as_cov2(cov)
    r = float(radius)
    if r <= 0.0:
        return PcResult(pc=0.0, method="grid", notes="radius<=0")
    if n_theta < 16 or n_r < 32:
        raise ValueError("n_theta and n_r too small")

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr = np.linspace(0.0, r, n_r)

    ct = np.cos(theta)
    st = np.sin(theta)
    xs = rr[:, None] * ct[None, :]
    ys = rr[:, None] * st[None, :]
    points = np.stack([xs, ys], axis=-1)

    pdf = _bvnorm_pdf(points, mu2, cov2)
    integrand = pdf * rr[:, None]

    dr = rr[1] - rr[0]
    dtheta = 2.0 * np.pi / n_theta
    pc = float(np.sum(integrand) * dr * dtheta)
    pc = max(0.0, min(pc, 1.0))
    return PcResult(pc=pc, method="grid")


def pc_circle_monte_carlo(
    mu: np.ndarray | list[float] | tuple[float, float],
    cov: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
    radius: float,
    *,
    n: int = 200_000,
    seed: Optional[int] = 0,
) -> PcResult:
    mu2 = _as_vec2(mu)
    cov2 = _as_cov2(cov)
    r = float(radius)
    if r <= 0.0:
        return PcResult(pc=0.0, method="mc", notes="radius<=0")
    if n < 10_000:
        raise ValueError("n too small")
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mu2, cov2, size=n, check_valid="raise")
    inside = np.sum(np.sum(samples * samples, axis=1) <= r * r)
    pc = float(inside / n)
    pc = max(0.0, min(pc, 1.0))
    return PcResult(pc=pc, method="mc", notes=f"n={n}")
