from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress


@dataclass
class SpectralResult:
    metrics: dict[str, float | int | str]
    top_left_vectors: np.ndarray | None


def _safe_entropy_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    total = singular_values.sum()
    if not np.isfinite(total) or total <= eps:
        return float("nan")
    p = singular_values / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def _mp_pdf_grid(aspect: float, num_grid: int = 4096) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return a numerical Marchenko--Pastur PDF grid for unit noise scale.

    `aspect` should be in (0, 1]. The support is [(1-sqrt(q))^2, (1+sqrt(q))^2].
    """
    q = float(np.clip(aspect, 1e-6, 1.0))
    lam_minus = (1.0 - np.sqrt(q)) ** 2
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    # Avoid exact endpoints where the density has numerical singularities.
    xs = np.linspace(lam_minus + 1e-8, lam_plus - 1e-8, num_grid)
    pdf = np.sqrt((lam_plus - xs) * (xs - lam_minus)) / (2.0 * np.pi * q * xs)
    pdf = np.maximum(pdf, 0.0)
    area = np.trapz(pdf, xs)
    if area > 0:
        pdf = pdf / area
    return xs, pdf, lam_minus, lam_plus


def _mp_quantile(aspect: float, quantile: float) -> float:
    xs, pdf, _, _ = _mp_pdf_grid(aspect)
    cdf = cumulative_trapezoid(pdf, xs, initial=0.0)
    cdf = cdf / max(cdf[-1], 1e-12)
    return float(np.interp(quantile, cdf, xs))


def mp_outlier_counts(
    eigenvalues: np.ndarray,
    rows: int,
    cols: int,
    edge_multipliers: list[float],
) -> dict[str, int | float]:
    """Heuristic MP-outlier counts from singular-value eigenvalues.

    We fit only a noise scale, using the median empirical eigenvalue matched to the MP median.
    This is intentionally reported with threshold sensitivity; it should be treated as a
    corroborative indicator, not a standalone boundary detector.
    """
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    if eig.size < 4:
        return {f"mp_outliers_x{m:g}": 0 for m in edge_multipliers} | {"mp_edge": float("nan")}

    aspect = min(rows, cols) / max(rows, cols)
    mp_median = _mp_quantile(aspect, 0.5)
    median_emp = float(np.median(eig))
    sigma2_hat = median_emp / max(mp_median, 1e-12)
    _, _, _, lam_plus = _mp_pdf_grid(aspect)
    edge = sigma2_hat * lam_plus

    out: dict[str, int | float] = {"mp_edge": float(edge)}
    for multiplier in edge_multipliers:
        out[f"mp_outliers_x{multiplier:g}"] = int(np.sum(eig > multiplier * edge))
    return out


def alpha_tail_fits(eigenvalues: np.ndarray, tail_fracs: list[float]) -> dict[str, float]:
    """Fit a simple log-rank/log-eigenvalue tail slope as a heavy-tail proxy.

    For eigenvalues sorted descending, a power-law tail has rank approximately proportional
    to lambda^{-alpha}. We estimate alpha from log(rank) vs log(lambda). This is a proxy used
    for sensitivity tracking, not a substitute for a full powerlaw package fit.
    """
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    eig = np.sort(eig)[::-1]
    n = eig.size
    out: dict[str, float] = {}
    if n < 8:
        for frac in tail_fracs:
            out[f"alpha_tail_frac_{frac:g}"] = float("nan")
        return out

    ranks = np.arange(1, n + 1, dtype=np.float64)
    for frac in tail_fracs:
        k = max(8, int(np.ceil(n * float(frac))))
        k = min(k, n)
        x = np.log(eig[:k])
        y = np.log(ranks[:k])
        if np.std(x) < 1e-12:
            alpha = float("nan")
        else:
            slope = linregress(x, y).slope
            alpha = float(-slope)
        out[f"alpha_tail_frac_{frac:g}"] = alpha
    return out


def subspace_cosine_mean(u_prev: np.ndarray | None, u_curr: np.ndarray | None) -> float:
    """Mean singular value of U_prev^T U_curr as a top-k subspace-stability score."""
    if u_prev is None or u_curr is None:
        return float("nan")
    k = min(u_prev.shape[1], u_curr.shape[1])
    if k == 0:
        return float("nan")
    a = u_prev[:, :k]
    b = u_curr[:, :k]
    if a.shape[0] != b.shape[0]:
        return float("nan")
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.mean(s))


def compute_spectral_indicators(
    weight: torch.Tensor,
    top_k_vectors: int = 8,
    mp_edge_multipliers: list[float] | None = None,
    alpha_tail_fracs: list[float] | None = None,
    center_weights: bool = False,
) -> SpectralResult:
    """Compute E1 spectral indicators for one 2D weight matrix.

    Returns scalar metrics plus top left singular vectors for checkpoint-to-checkpoint
    stability calculations.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={tuple(weight.shape)}")
    mp_edge_multipliers = mp_edge_multipliers or [1.0, 1.1, 1.25]
    alpha_tail_fracs = alpha_tail_fracs or [0.2, 0.3, 0.5]

    w = weight.detach().float()
    if center_weights:
        w = w - w.mean()
    rows, cols = int(w.shape[0]), int(w.shape[1])

    # Full SVD is acceptable for the initial 70M/160M/410M E1 scope and gives U for stability.
    # If larger models are added, this function can be swapped for randomized SVD.
    u, s, _ = torch.linalg.svd(w, full_matrices=False)
    s_np = s.detach().cpu().numpy().astype(np.float64)
    eig_np = s_np**2

    fro_norm = float(np.sqrt(np.sum(eig_np)))
    spectral_norm = float(s_np[0]) if s_np.size else float("nan")
    stable_rank = float((fro_norm**2) / max(spectral_norm**2, 1e-12))
    effective_rank = _safe_entropy_rank(s_np)

    metrics: dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "num_singular_values": int(s_np.size),
        "frobenius_norm": fro_norm,
        "spectral_norm": spectral_norm,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "nuclear_norm": float(np.sum(s_np)),
        "mean_singular_value": float(np.mean(s_np)) if s_np.size else float("nan"),
        "median_eigenvalue": float(np.median(eig_np)) if eig_np.size else float("nan"),
    }
    metrics.update(mp_outlier_counts(eig_np, rows, cols, mp_edge_multipliers))
    metrics.update(alpha_tail_fits(eig_np, alpha_tail_fracs))

    k = min(int(top_k_vectors), u.shape[1])
    top_u = u[:, :k].detach().cpu().numpy().astype(np.float32) if k > 0 else None
    return SpectralResult(metrics=metrics, top_left_vectors=top_u)
