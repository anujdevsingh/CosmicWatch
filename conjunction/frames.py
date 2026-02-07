from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


@dataclass(frozen=True)
class Basis3:
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.stack([self.e1, self.e2, self.e3], axis=1)


@dataclass(frozen=True)
class EncounterPlane:
    along: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray

    def plane_matrix(self) -> np.ndarray:
        return np.stack([self.axis1, self.axis2], axis=1)


def rtn_basis(r_eci: np.ndarray, v_eci: np.ndarray) -> Basis3:
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    r_hat = _unit(r)
    n_hat = _unit(np.cross(r, v))
    t_hat = _unit(np.cross(n_hat, r_hat))
    return Basis3(e1=r_hat, e2=t_hat, e3=n_hat)


def cov_rtn_to_eci(cov_rtn: np.ndarray, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov_rtn, dtype=float).reshape(3, 3)
    cov = (cov + cov.T) / 2.0
    q = rtn_basis(r_eci, v_eci).as_matrix()
    return q @ cov @ q.T


def encounter_plane(r_rel_eci: np.ndarray, v_rel_eci: np.ndarray) -> EncounterPlane:
    r_rel = np.asarray(r_rel_eci, dtype=float).reshape(3)
    v_rel = np.asarray(v_rel_eci, dtype=float).reshape(3)
    along = _unit(v_rel)
    normal = _unit(np.cross(r_rel, v_rel))
    axis1 = _unit(np.cross(normal, along))
    axis2 = normal
    return EncounterPlane(along=along, axis1=axis1, axis2=axis2)


def project_to_encounter_plane(vec_eci: np.ndarray, plane: EncounterPlane) -> np.ndarray:
    v = np.asarray(vec_eci, dtype=float).reshape(3)
    return np.array([float(plane.axis1 @ v), float(plane.axis2 @ v)], dtype=float)


def project_cov_to_encounter_plane(cov_eci: np.ndarray, plane: EncounterPlane) -> np.ndarray:
    cov = np.asarray(cov_eci, dtype=float).reshape(3, 3)
    cov = (cov + cov.T) / 2.0
    a = plane.plane_matrix()
    cov2 = a.T @ cov @ a
    cov2 = (cov2 + cov2.T) / 2.0
    return cov2


def combined_pos_cov_encounter_plane(
    primary_r_eci: np.ndarray,
    primary_v_eci: np.ndarray,
    secondary_r_eci: np.ndarray,
    secondary_v_eci: np.ndarray,
    cov1_rtn_pos: np.ndarray,
    cov2_rtn_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r1 = np.asarray(primary_r_eci, dtype=float).reshape(3)
    v1 = np.asarray(primary_v_eci, dtype=float).reshape(3)
    r2 = np.asarray(secondary_r_eci, dtype=float).reshape(3)
    v2 = np.asarray(secondary_v_eci, dtype=float).reshape(3)
    r_rel = r2 - r1
    v_rel = v2 - v1

    p1_eci = cov_rtn_to_eci(cov1_rtn_pos, r1, v1)
    p2_eci = cov_rtn_to_eci(cov2_rtn_pos, r2, v2)
    p_rel = p1_eci + p2_eci

    plane = encounter_plane(r_rel, v_rel)
    mu2 = project_to_encounter_plane(r_rel, plane)
    cov2 = project_cov_to_encounter_plane(p_rel, plane)
    return mu2, cov2
