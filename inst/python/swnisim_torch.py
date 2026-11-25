"""Batched PyTorch GPU/CPU simulation backend for swNIsim.

Implements exact batched mixed-model solvers matching lme4::lmer (ML and REML,
with cell precision weights K) and lme4::glmer(..., family = binomial(), nAGQ = 0)
for repeated cross-sectional stepped-wedge non-inferiority trials with a cluster
random intercept (1 | cluster).
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_PHI_RATIO = (math.sqrt(5.0) - 1.0) / 2.0  # 0.6180339887498949


def _get_cuda_ext_safe() -> Optional[Any]:
    try:
        pkg_dir = os.path.dirname(__file__)
        if pkg_dir and pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        import swnisim_cuda
        return swnisim_cuda.get_cuda_ext()
    except Exception:
        return None


def probe_torch() -> Dict[str, Any]:
    """Return PyTorch, GPU, and fused CUDA kernel availability metadata."""
    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(
         hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    gpu_name = ""
    if cuda_ok:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = "CUDA GPU"
    elif mps_ok:
        gpu_name = "Apple MPS GPU"

    if cuda_ok:
        rec_dev = "cuda"
    elif mps_ok:
        rec_dev = "mps"
    else:
        rec_dev = "cpu"

    nvcc_ok = False
    try:
        pkg_dir = os.path.dirname(__file__)
        if pkg_dir and pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        import swnisim_cuda
        nvcc_ok = bool(swnisim_cuda.cuda_kernel_info().get("nvcc_detected", False))
    except Exception:
        nvcc_ok = False

    return {
        "available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_ok,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_ok else 0,
        "mps_available": mps_ok,
        "device_name": gpu_name,
        "recommended_device": rec_dev,
        "cuda_kernel_supported": bool(cuda_ok and nvcc_ok),
    }


def _resolve_device(device: Optional[str] = None) -> Tuple[torch.device, torch.dtype, str]:
    if device is None or device == "" or device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU requested, but torch.cuda.is_available() is False. "
            "In Google Colab, enable a GPU via Runtime -> Change runtime type -> T4 GPU."
        )
    if dev.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS GPU requested, but torch.backends.mps.is_available() is False.")

    dtype = torch.float32 if dev.type == "mps" else torch.float64
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        label = f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    else:
        label = str(dev)
    return dev, dtype, label


def _make_generator(dev: torch.device, seed: int) -> torch.Generator:
    gen_dev = "cuda" if dev.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_dev)
    gen.manual_seed(int(seed))
    return gen


def _randn(
    shape: Tuple[int, ...],
    gen: torch.Generator,
    dev: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dev.type == "mps":
        return torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32).to(dev)
    return torch.randn(shape, generator=gen, device=dev, dtype=dtype)


def _sample_binomial(
    total_count: torch.Tensor,
    probs: torch.Tensor,
    gen: torch.Generator,
    dev: torch.device,
) -> torch.Tensor:
    probs_clamped = torch.clamp(probs, min=0.0, max=1.0)
    if dev.type == "mps":
        c_cpu = total_count.to("cpu", dtype=torch.float64)
        p_cpu = probs_clamped.to("cpu", dtype=torch.float64)
        draws = torch.binomial(c_cpu, p_cpu, generator=gen)
        return draws.to(dev, dtype=probs.dtype)
    try:
        return torch.binomial(total_count, probs_clamped, generator=gen)
    except Exception:
        c_cpu = total_count.to("cpu", dtype=torch.float64)
        p_cpu = probs_clamped.to("cpu", dtype=torch.float64)
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(gen.initial_seed()))
        return torch.binomial(c_cpu, p_cpu, generator=cpu_gen).to(dev, dtype=probs.dtype)


# ---------------------------------------------------------------------------
# Continuous LMM Engine (matches lme4::lmer with weights = K, REML = TRUE/FALSE)
# ---------------------------------------------------------------------------

def _eval_continuous_batch(
    theta: torch.Tensor,        # [B]
    M_within: torch.Tensor,     # [p, p]
    S_X: torch.Tensor,          # [I, p]
    W_clus: torch.Tensor,       # [I]
    s_y: torch.Tensor,          # [B, I]
    q_within: torch.Tensor,     # [B]
    m_within: torch.Tensor,     # [B, p]
    N_cells: int,
    p_dim: int,
    reml: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate exact profiled REML/ML deviance, beta_tx, SE_tx, and validity at theta [B]."""
    lam = theta * theta  # [B]
    lam_W = lam.unsqueeze(1) * W_clus.unsqueeze(0)  # [B, I]
    inv_d = 1.0 / (1.0 + lam_W)  # [B, I]
    scale_i = inv_d / W_clus.unsqueeze(0)  # [B, I]

    # A[b] = M_within + S_X^T diag(scale_i[b]) S_X -> [B, p, p]
    scaled_SX = scale_i.unsqueeze(2) * S_X.unsqueeze(0)  # [B, I, p]
    A = M_within.unsqueeze(0) + torch.einsum("ip,biq->bpq", S_X, scaled_SX)

    # rhs[b] = m_within[b] + S_X^T (scale_i[b] * s_y[b]) -> [B, p]
    rhs = m_within + torch.einsum("ip,bi->bp", S_X, scale_i * s_y)

    L, info = torch.linalg.cholesky_ex(A)
    bad = info != 0
    if bool(bad.any().item()):
        eye_p = torch.eye(p_dim, device=A.device, dtype=A.dtype).unsqueeze(0)
        L = torch.where(bad.view(-1, 1, 1), eye_p, L)
    beta = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [B, p]

    # r2[b] = q_within[b] + sum_i (inv_d[b, i] * s_y[b, i]^2 / W_i) - beta[b]^T rhs[b]
    tiny = torch.finfo(A.dtype).tiny
    s_y2_over_W = (s_y * s_y) / W_clus.unsqueeze(0)
    r2 = q_within + (inv_d * s_y2_over_W).sum(dim=1) - (beta * rhs).sum(dim=1)
    r2_safe = torch.clamp(r2, min=tiny)

    logdet_V = torch.log1p(lam_W).sum(dim=1)
    df = float(N_cells - p_dim) if reml else float(N_cells)
    if reml:
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
        logdet_A = 2.0 * torch.log(torch.clamp(diag_L, min=tiny)).sum(dim=1)
        dev = logdet_V + logdet_A + df * torch.log(r2_safe)
    else:
        dev = logdet_V + df * torch.log(r2_safe)

    sigma2 = r2 / df
    l_last = L[:, -1, -1]
    se_tx = torch.sqrt(torch.clamp(sigma2, min=0.0)) / torch.clamp(l_last, min=tiny)
    beta_tx = beta[:, -1]

    valid = (
        (info == 0)
        & (r2 > 0.0)
        & torch.isfinite(dev)
        & torch.isfinite(beta_tx)
        & torch.isfinite(se_tx)
        & (se_tx > 0.0)
    )
    dev = torch.where(valid, dev, torch.full_like(dev, float("inf")))
    return dev, beta_tx, se_tx, valid, r2


def _fit_continuous_from_y(
    y_bit: torch.Tensor,        # [B, I, Tp]
    X_itp: torch.Tensor,        # [I, Tp, p]
    K_it: torch.Tensor,         # [I, Tp]
    precomp: Dict[str, Any],
    reml: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit batched continuous LMM on y_bit [B, I, Tp]. Returns (theta_hat, beta_tx, se_tx, valid)."""
    M_within = precomp["M_within"]
    S_X = precomp["S_X"]
    W_clus = precomp["W_clus"]
    u_grid = precomp["u_grid"]
    s0 = precomp["s0"]
    A_grid_inv = precomp["A_grid_inv"]      # [G, p, p]
    const_grid = precomp["const_grid"]      # [G]
    scale_grid = precomp["scale_grid"]      # [G, I]
    inv_d_grid = precomp["inv_d_grid"]      # [G, I]
    N_cells = precomp["N_cells"]
    p_dim = precomp["p_dim"]

    wy = y_bit * K_it.unsqueeze(0)  # [B, I, Tp]
    s_y = wy.sum(dim=2)             # [B, I]
    s_y2_over_W = (s_y * s_y) / W_clus.unsqueeze(0)  # [B, I]
    q_within = (wy * y_bit).sum(dim=(1, 2)) - s_y2_over_W.sum(dim=1)  # [B]
    m_Xy = torch.einsum("bit,itp->bp", wy, X_itp)  # [B, p]
    m_within = m_Xy - torch.einsum("bi,ip->bp", s_y / W_clus.unsqueeze(0), S_X)  # [B, p]

    # 1. Fast vectorized coarse grid evaluation across all G grid points (no Cholesky per trial!)
    # rhs_grid[b, g, p] = m_within[b, p] + sum_i scale_grid[g, i] * s_y[b, i] * S_X[i, p]
    SX_sy = torch.einsum("bi,ip->bip", s_y, S_X)  # [B, I, p]
    rhs_grid = m_within.unsqueeze(1) + torch.einsum("gi,bip->bgp", scale_grid, SX_sy)  # [B, G, p]
    beta_grid = torch.einsum("gpq,bgq->bgp", A_grid_inv, rhs_grid)  # [B, G, p]
    r2_grid = (
        q_within.unsqueeze(1)
        + torch.einsum("gi,bi->bg", inv_d_grid, s_y2_over_W)
        - (beta_grid * rhs_grid).sum(dim=2)
    )
    tiny = torch.finfo(y_bit.dtype).tiny
    df = float(N_cells - p_dim) if reml else float(N_cells)
    dev_grid = const_grid.unsqueeze(0) + df * torch.log(torch.clamp(r2_grid, min=tiny))  # [B, G]

    G = u_grid.shape[0]
    best_idx = torch.argmin(dev_grid, dim=1)  # [B]
    left_idx = torch.clamp(best_idx - 1, min=0)
    right_idx = torch.clamp(best_idx + 1, max=G - 1)
    a = u_grid[left_idx]
    b = u_grid[right_idx]

    # 2. Batched golden-section refinement on u in [a, b], theta = s0 * u / (1 - u)
    x1 = b - _PHI_RATIO * (b - a)
    x2 = a + _PHI_RATIO * (b - a)
    f1, _, _, _, _ = _eval_continuous_batch(
        s0 * x1 / (1.0 - x1), M_within, S_X, W_clus, s_y, q_within, m_within, N_cells, p_dim, reml
    )
    f2, _, _, _, _ = _eval_continuous_batch(
        s0 * x2 / (1.0 - x2), M_within, S_X, W_clus, s_y, q_within, m_within, N_cells, p_dim, reml
    )

    for _ in range(22):
        go_left = f1 <= f2
        b = torch.where(go_left, x2, b)
        a = torch.where(go_left, a, x1)
        x_new_left = b - _PHI_RATIO * (b - a)
        x_new_right = a + _PHI_RATIO * (b - a)

        # Reuse one interior evaluation per step!
        x2_next = torch.where(go_left, x1, x_new_right)
        f2_next = f1
        x1_next = torch.where(go_left, x_new_left, x2)
        f1_next = f2

        x_eval = torch.where(go_left, x_new_left, x_new_right)
        f_eval, _, _, _, _ = _eval_continuous_batch(
            s0 * x_eval / (1.0 - x_eval),
            M_within, S_X, W_clus, s_y, q_within, m_within, N_cells, p_dim, reml,
        )
        x1 = x1_next
        x2 = x2_next
        f1 = torch.where(go_left, f_eval, f1_next)
        f2 = torch.where(go_left, f2_next, f_eval)

    u_mid = 0.5 * (a + b)
    theta_cand = s0 * u_mid / (1.0 - u_mid)
    dev_cand, beta_cand, se_cand, valid_cand, _ = _eval_continuous_batch(
        theta_cand, M_within, S_X, W_clus, s_y, q_within, m_within, N_cells, p_dim, reml
    )

    # 3. Exact boundary check at theta = 0 (matches check.conv.singular = "ignore")
    theta_zero = torch.zeros_like(theta_cand)
    dev_zero, beta_zero, se_zero, valid_zero, _ = _eval_continuous_batch(
        theta_zero, M_within, S_X, W_clus, s_y, q_within, m_within, N_cells, p_dim, reml
    )

    use_zero = valid_zero & ((~valid_cand) | (dev_zero <= dev_cand))
    theta_hat = torch.where(use_zero, theta_zero, theta_cand)
    beta_tx = torch.where(use_zero, beta_zero, beta_cand)
    se_tx = torch.where(use_zero, se_zero, se_cand)
    valid = valid_zero | valid_cand
    return theta_hat, beta_tx, se_tx, valid


def _prepare_continuous_design(
    X_itp: torch.Tensor,
    K_it: torch.Tensor,
    sdw: float,
    sdc: float,
    reml: bool,
) -> Optional[Dict[str, Any]]:
    dev = X_itp.device
    dtype = X_itp.dtype
    I_c, Tp, p_dim = X_itp.shape
    N_cells = I_c * Tp
    if reml and N_cells <= p_dim:
        return None

    WX = K_it.unsqueeze(2) * X_itp  # [I, Tp, p]
    M_XX = torch.einsum("itp,itq->pq", X_itp, WX)
    _, info0 = torch.linalg.cholesky_ex(M_XX)
    if int(info0.item()) != 0:
        return None

    S_X = WX.sum(dim=1)  # [I, p]
    W_clus = K_it.sum(dim=1)  # [I]
    M_within = M_XX - torch.einsum("ip,iq->pq", S_X, S_X / W_clus.unsqueeze(1))

    s0 = max(0.5, float(sdc) / max(float(sdw), 1e-3))
    G = 64
    u_grid = torch.linspace(0.0, 0.996, G, device=dev, dtype=dtype)
    theta_grid = s0 * u_grid / (1.0 - u_grid)  # [G]
    lam_W = (theta_grid * theta_grid).unsqueeze(1) * W_clus.unsqueeze(0)  # [G, I]
    inv_d_grid = 1.0 / (1.0 + lam_W)  # [G, I]
    scale_grid = inv_d_grid / W_clus.unsqueeze(0)  # [G, I]

    A_grid = M_within.unsqueeze(0) + torch.einsum(
        "ip,giq->gpq", S_X, scale_grid.unsqueeze(2) * S_X.unsqueeze(0)
    )
    L_grid, info_g = torch.linalg.cholesky_ex(A_grid)
    if bool((info_g != 0).any().item()):
        return None
    eye_p = torch.eye(p_dim, device=dev, dtype=dtype).unsqueeze(0).expand(G, p_dim, p_dim)
    A_grid_inv = torch.cholesky_solve(eye_p, L_grid)

    logdet_V_grid = torch.log1p(lam_W).sum(dim=1)  # [G]
    if reml:
        diag_Lg = torch.diagonal(L_grid, dim1=-2, dim2=-1)
        logdet_A_grid = 2.0 * torch.log(diag_Lg).sum(dim=1)
        const_grid = logdet_V_grid + logdet_A_grid
    else:
        const_grid = logdet_V_grid

    return {
        "M_within": M_within,
        "S_X": S_X,
        "W_clus": W_clus,
        "u_grid": u_grid,
        "s0": s0,
        "A_grid_inv": A_grid_inv,
        "const_grid": const_grid,
        "scale_grid": scale_grid,
        "inv_d_grid": inv_d_grid,
        "N_cells": N_cells,
        "p_dim": p_dim,
    }


# ---------------------------------------------------------------------------
# Binary GLMM Engine (matches lme4::glmer(..., family = binomial(), nAGQ = 0))
# ---------------------------------------------------------------------------

def _pirls_binary_batch(
    theta: torch.Tensor,          # [B]
    y_bit: torch.Tensor,          # [B, I, Tp]
    K_it: torch.Tensor,           # [I, Tp]
    tx_it: torch.Tensor,          # [I, Tp]
    tx2_it: torch.Tensor,         # [I, Tp]
    adjust_period: bool,
    beta_init: torch.Tensor,      # [B, p]
    z_init: torch.Tensor,         # [B, I]
    max_iter: int = 25,
    tol: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run joint PIRLS over (z, beta) for nAGQ = 0 binomial GLMM on a batch of B trials."""
    B, I_c, Tp = y_bit.shape
    p_dim = (Tp + 1) if adjust_period else 2
    dtype = y_bit.dtype
    dev = y_bit.device

    beta = beta_init.clone()
    z = z_init.clone()
    th2 = theta * theta  # [B]
    active = torch.ones(B, dtype=torch.bool, device=dev)
    failed = torch.zeros(B, dtype=torch.bool, device=dev)

    K_b = K_it.unsqueeze(0)      # [1, I, Tp]
    tx_b = tx_it.unsqueeze(0)    # [1, I, Tp]
    tx2_b = tx2_it.unsqueeze(0)  # [1, I, Tp]

    def _compute_eta(b_vec: torch.Tensor, z_vec: torch.Tensor) -> torch.Tensor:
        if adjust_period:
            # b_vec[:, 0] = intercept, b_vec[:, 1:Tp] = period 2..Tp, b_vec[:, Tp] = delayed_tx
            eta_f = b_vec[:, 0:1].unsqueeze(2) + b_vec[:, -1:].unsqueeze(2) * tx_b
            if Tp > 1:
                eta_f = eta_f.clone()
                eta_f[:, :, 1:] = eta_f[:, :, 1:] + b_vec[:, 1:Tp].unsqueeze(1)
        else:
            eta_f = b_vec[:, 0:1].unsqueeze(2) + b_vec[:, 1:2].unsqueeze(2) * tx_b
        eta_raw = eta_f + (theta.unsqueeze(1) * z_vec).unsqueeze(2)
        return torch.clamp(eta_raw, min=-30.0, max=30.0)

    def _build_schur(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_clus = W.sum(dim=2)  # [B, I]
        D_z = 1.0 + th2.unsqueeze(1) * W_clus  # [B, I]
        W_tx_it = W * tx_b  # [B, I, Tp]
        if adjust_period:
            # S_WX: [B, I, Tp + 1]
            S_WX = torch.empty((B, I_c, p_dim), device=dev, dtype=dtype)
            S_WX[:, :, 0] = W_clus
            if Tp > 1:
                S_WX[:, :, 1:Tp] = W[:, :, 1:]
            S_WX[:, :, -1] = W_tx_it.sum(dim=2)

            # X^T W X: [B, Tp + 1, Tp + 1] exploiting exact diagonal period block!
            W_per = W.sum(dim=1)          # [B, Tp]
            W_tx_per = W_tx_it.sum(dim=1) # [B, Tp]
            W_tx2 = (W * tx2_b).sum(dim=(1, 2))  # [B]

            XTWX = torch.zeros((B, p_dim, p_dim), device=dev, dtype=dtype)
            XTWX[:, 0, 0] = W_per.sum(dim=1)
            if Tp > 1:
                w_p_sub = W_per[:, 1:]
                idx_p = torch.arange(1, Tp, device=dev)
                XTWX[:, 0, 1:Tp] = w_p_sub
                XTWX[:, 1:Tp, 0] = w_p_sub
                XTWX[:, idx_p, idx_p] = w_p_sub
                w_tx_sub = W_tx_per[:, 1:]
                XTWX[:, 1:Tp, -1] = w_tx_sub
                XTWX[:, -1, 1:Tp] = w_tx_sub
            w_tx_tot = W_tx_per.sum(dim=1)
            XTWX[:, 0, -1] = w_tx_tot
            XTWX[:, -1, 0] = w_tx_tot
            XTWX[:, -1, -1] = W_tx2
        else:
            S_WX = torch.stack([W_clus, W_tx_it.sum(dim=2)], dim=2)  # [B, I, 2]
            w00 = W_clus.sum(dim=1)
            w01 = W_tx_it.sum(dim=(1, 2))
            w11 = (W * tx2_b).sum(dim=(1, 2))
            XTWX = torch.empty((B, 2, 2), device=dev, dtype=dtype)
            XTWX[:, 0, 0] = w00
            XTWX[:, 0, 1] = w01
            XTWX[:, 1, 0] = w01
            XTWX[:, 1, 1] = w11

        # Schur complement: A_b = X^T W X - theta^2 S_WX^T diag(1 / D_z) S_WX
        scale_z = (th2.unsqueeze(1) / D_z).unsqueeze(2)  # [B, I, 1]
        A_b = XTWX - torch.einsum("bip,biq->bpq", S_WX, scale_z * S_WX)
        return A_b, S_WX, D_z

    def _penalized_dev(eta_v: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        # -2 * sum [ y * eta - K * (max(eta, 0) + log1p(exp(-|eta|))) ] + ||z||^2
        log1p_exp = torch.clamp(eta_v, min=0.0) + torch.log1p(torch.exp(-torch.abs(eta_v)))
        ll = (y_bit * eta_v - K_b * log1p_exp).sum(dim=(1, 2))
        return -2.0 * ll + (z_v * z_v).sum(dim=1)

    eta = _compute_eta(beta, z)
    p_dev = _penalized_dev(eta, z)

    for _ in range(max_iter):
        if not bool(active.any().item()):
            break
        mu_p = torch.sigmoid(eta)
        W = torch.clamp(K_b * mu_p * (1.0 - mu_p), min=1e-15)
        g_eta = y_bit - K_b * mu_p  # [B, I, Tp]

        A_b, S_WX, D_z = _build_schur(W)
        s_g = g_eta.sum(dim=2)  # [B, I]
        rhs_z = theta.unsqueeze(1) * s_g - z  # [B, I]

        if adjust_period:
            g_per = g_eta.sum(dim=1)  # [B, Tp]
            rhs_b = torch.empty((B, p_dim), device=dev, dtype=dtype)
            rhs_b[:, 0] = g_per.sum(dim=1)
            if Tp > 1:
                rhs_b[:, 1:Tp] = g_per[:, 1:]
            rhs_b[:, -1] = (g_eta * tx_b).sum(dim=(1, 2))
        else:
            rhs_b = torch.stack(
                [g_eta.sum(dim=(1, 2)), (g_eta * tx_b).sum(dim=(1, 2))], dim=1
            )

        b_b = rhs_b - theta.unsqueeze(1) * torch.einsum(
            "bip,bi->bp", S_WX, rhs_z / D_z
        )
        L_b, info_b = torch.linalg.cholesky_ex(A_b)
        bad_chol = active & (info_b != 0)
        if bool(bad_chol.any().item()):
            failed = failed | bad_chol
            active = active & (~bad_chol)
            if not bool(active.any().item()):
                break
            # Replace failed diagonal with identity so cholesky_solve doesn't NaN
            eye_p = torch.eye(p_dim, device=dev, dtype=dtype).unsqueeze(0)
            L_b = torch.where(bad_chol.view(B, 1, 1), eye_p, L_b)

        dbeta = torch.cholesky_solve(b_b.unsqueeze(-1), L_b).squeeze(-1)  # [B, p]
        SWX_db = torch.einsum("bip,bp->bi", S_WX, dbeta)
        dz = (rhs_z - theta.unsqueeze(1) * SWX_db) / D_z

        ndec2 = torch.clamp(
            (rhs_z * dz).sum(dim=1) + (rhs_b * dbeta).sum(dim=1), min=0.0
        )
        ndec = torch.sqrt(ndec2)

        # Vectorized step-halving if penalized deviance increases
        step = torch.where(active, torch.ones_like(theta), torch.zeros_like(theta))
        beta_try = beta + step.unsqueeze(1) * dbeta
        z_try = z + step.unsqueeze(1) * dz
        eta_try = _compute_eta(beta_try, z_try)
        pdev_try = _penalized_dev(eta_try, z_try)

        for _ in range(4):
            need_halve = active & (pdev_try > p_dev + 1e-6 * (1.0 + torch.abs(p_dev)))
            if not bool(need_halve.any().item()):
                break
            step = torch.where(need_halve, 0.5 * step, step)
            beta_try = beta + step.unsqueeze(1) * dbeta
            z_try = z + step.unsqueeze(1) * dz
            eta_try = _compute_eta(beta_try, z_try)
            pdev_try = _penalized_dev(eta_try, z_try)

        beta = torch.where(active.unsqueeze(1), beta_try, beta)
        z = torch.where(active.unsqueeze(1), z_try, z)
        eta = torch.where(active.view(B, 1, 1), eta_try, eta)
        p_dev = torch.where(active, pdev_try, p_dev)

        converged_now = ndec < tol
        active = active & (~converged_now)

    # Final Laplace deviance & SE at converged mode
    tiny = torch.finfo(dtype).tiny
    mu_p = torch.sigmoid(eta)
    W = torch.clamp(K_b * mu_p * (1.0 - mu_p), min=1e-15)
    A_b, _, D_z = _build_schur(W)
    L_b, info_b = torch.linalg.cholesky_ex(A_b)

    laplace_dev = p_dev + torch.log(torch.clamp(D_z, min=tiny)).sum(dim=1)
    l_last = L_b[:, -1, -1]
    se_tx = 1.0 / torch.clamp(l_last, min=tiny)
    beta_tx = beta[:, -1]

    valid = (
        (~failed)
        & (info_b == 0)
        & torch.isfinite(laplace_dev)
        & torch.isfinite(beta_tx)
        & torch.isfinite(se_tx)
        & (se_tx > 0.0)
    )
    laplace_dev = torch.where(valid, laplace_dev, torch.full_like(laplace_dev, float("inf")))
    return laplace_dev, beta_tx, se_tx, valid, beta, z


def _fit_binary_from_y(
    y_bit: torch.Tensor,          # [B, I, Tp]
    K_it: torch.Tensor,           # [I, Tp]
    tx_it: torch.Tensor,          # [I, Tp]
    adjust_period: bool,
    sdc: float,
    cuda_ext: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit batched nAGQ = 0 binomial GLMM on y_bit [B, I, Tp]."""
    B, I_c, Tp = y_bit.shape
    p_dim = (Tp + 1) if adjust_period else 2
    dev = y_bit.device
    dtype = y_bit.dtype
    tx2_it = tx_it * tx_it

    use_cuda_pirls = (
        cuda_ext is not None
        and dev.type == "cuda"
        and 2 <= Tp <= 8
        and I_c <= 2048
        and (p_dim == 2 or p_dim == Tp + 1)
    )
    y_c = y_bit.contiguous()
    K_c = K_it.contiguous()
    tx_c = tx_it.contiguous()
    all_idx = torch.arange(B, device=dev, dtype=torch.int64)

    def _run_pirls(
        th_sub: torch.Tensor,
        idx_sub: torch.Tensor,
        b_in: torch.Tensor,
        z_in: torch.Tensor,
        max_iter: int,
        tol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_cuda_pirls:
            res = cuda_ext.pirls_binary_cuda(
                th_sub, idx_sub, y_c, K_c, tx_c, b_in, z_in, bool(adjust_period), int(max_iter), float(tol)
            )
            return res[0], res[1], res[2], res[3], res[4], res[5]
        y_sub = y_c if idx_sub.shape[0] == B and idx_sub is all_idx else y_c[idx_sub]
        return _pirls_binary_batch(
            th_sub, y_sub, K_c, tx_c, tx2_it, adjust_period, b_in, z_in, max_iter=max_iter, tol=tol
        )

    # Initial beta0 from empirical logit per trial
    tot_y = y_c.sum(dim=(1, 2))
    tot_K = float(K_c.sum().item())
    p_emp = (tot_y + 0.5) / (tot_K + 1.0)
    beta_0_init = torch.zeros((B, p_dim), device=dev, dtype=dtype)
    beta_0_init[:, 0] = torch.log(p_emp / (1.0 - p_emp))
    z_0_init = torch.zeros((B, I_c), device=dev, dtype=dtype)

    # 1. Exact evaluation at boundary theta = 0 (also warm-starts the coarse grid)
    theta_zero = torch.zeros(B, device=dev, dtype=dtype)
    dev_zero, beta_tx_zero, se_tx_zero, valid_zero, beta_ws, z_ws = _run_pirls(
        theta_zero, all_idx, beta_0_init, z_0_init, max_iter=25, tol=1e-7
    )

    # 2. Sequential warm-started 8-point coarse bracket over u in [0, 0.95]
    s0 = max(0.5, float(sdc))
    G = 8
    u_grid = torch.linspace(0.0, 0.95, G, device=dev, dtype=dtype)
    dev_grid = torch.empty((B, G), device=dev, dtype=dtype)
    dev_grid[:, 0] = dev_zero

    beta_cur = beta_ws
    z_cur = z_ws
    beta_best = beta_ws.clone()
    z_best = z_ws.clone()
    best_dev_so_far = dev_zero.clone()

    for g in range(1, G):
        th_g = torch.full((B,), float((s0 * u_grid[g] / (1.0 - u_grid[g])).item()), device=dev, dtype=dtype)
        dg, _, _, vg, b_next, z_next = _run_pirls(
            th_g, all_idx, beta_cur, z_cur, max_iter=18, tol=1e-6
        )
        dev_grid[:, g] = dg
        beta_cur = torch.where(vg.unsqueeze(1), b_next, beta_cur)
        z_cur = torch.where(vg.unsqueeze(1), z_next, z_cur)
        improved = vg & (dg < best_dev_so_far)
        best_dev_so_far = torch.where(improved, dg, best_dev_so_far)
        beta_best = torch.where(improved.unsqueeze(1), b_next, beta_best)
        z_best = torch.where(improved.unsqueeze(1), z_next, z_best)

    best_idx = torch.argmin(dev_grid, dim=1)
    left_idx = torch.clamp(best_idx - 1, min=0)
    right_idx = torch.clamp(best_idx + 1, max=G - 1)
    a = u_grid[left_idx]
    b = u_grid[right_idx]

    # 3. Batched safeguarded Brent/parabolic refinement with active-set compaction
    # Maintain bracket [a, b] and best interior point (x, fx) plus two prior points (w, fw), (v, fv)
    x = u_grid[best_idx].clone()
    fx = best_dev_so_far.clone()
    # Handle boundary best_idx == 0 or G-1 by starting x at interior golden point
    at_edge = (best_idx == 0) | (best_idx == G - 1)
    x_init_mid = a + (1.0 - _PHI_RATIO) * (b - a)
    x = torch.where(at_edge, x_init_mid, x)
    if bool(at_edge.any().item()):
        edge_idx = torch.nonzero(at_edge, as_tuple=False).reshape(-1)
        f_edge, _, _, v_edge, b_edge, z_edge = _run_pirls(
            s0 * x[edge_idx] / (1.0 - x[edge_idx]),
            edge_idx, beta_best[edge_idx], z_best[edge_idx], max_iter=16, tol=1e-7,
        )
        fx[edge_idx] = f_edge
        beta_best[edge_idx] = torch.where(v_edge.unsqueeze(1), b_edge, beta_best[edge_idx])
        z_best[edge_idx] = torch.where(v_edge.unsqueeze(1), z_edge, z_best[edge_idx])

    w = x.clone()
    fw = fx.clone()
    v = x.clone()
    fv = fx.clone()
    beta_ws = beta_best.clone()
    z_ws = z_best.clone()

    u_tol = 1.5e-5
    active = (b - a) > (2.0 * u_tol)

    for _ in range(22):
        idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            break

        aa = a[idx]
        bb = b[idx]
        xx = x[idx]
        fxx = fx[idx]
        ww = w[idx]
        fww = fw[idx]
        vv = v[idx]
        fvv = fv[idx]

        mid = 0.5 * (aa + bb)
        # Parabolic interpolation through (v, fv), (w, fw), (x, fx)
        r = (xx - ww) * (fxx - fvv)
        q = (xx - vv) * (fxx - fww)
        p_num = (xx - vv) * q - (xx - ww) * r
        q_den = 2.0 * (q - r)
        p_num = torch.where(q_den > 0.0, -p_num, p_num)
        q_den = torch.abs(q_den)

        # Safe margin inside (aa, bb)
        margin_ab = 0.02 * (bb - aa)
        can_para = (
            (q_den > 1e-18)
            & torch.isfinite(p_num)
            & torch.isfinite(q_den)
            & (xx + p_num / torch.clamp(q_den, min=1e-18) > aa + margin_ab)
            & (xx + p_num / torch.clamp(q_den, min=1e-18) < bb - margin_ab)
            & (torch.abs(p_num / torch.clamp(q_den, min=1e-18)) > 0.5 * u_tol)
        )
        u_para = xx + p_num / torch.clamp(q_den, min=1e-18)

        # Golden-section fallback on larger sub-interval
        go_right_seg = xx < mid
        u_gold = xx + (1.0 - _PHI_RATIO) * torch.where(go_right_seg, bb - xx, aa - xx)
        u_try = torch.where(can_para, u_para, u_gold)

        f_try, _, _, v_try, b_try, z_try = _run_pirls(
            s0 * u_try / (1.0 - u_try),
            idx, beta_ws[idx], z_ws[idx], max_iter=15, tol=1e-8,
        )
        beta_ws[idx] = torch.where(v_try.unsqueeze(1), b_try, beta_ws[idx])
        z_ws[idx] = torch.where(v_try.unsqueeze(1), z_try, z_ws[idx])

        better = f_try <= fxx
        # Update bracket [a, b]
        a_new = torch.where(better, torch.where(u_try >= xx, xx, aa), torch.where(u_try < xx, u_try, aa))
        b_new = torch.where(better, torch.where(u_try < xx, xx, bb), torch.where(u_try >= xx, u_try, bb))

        # Update (v, fv), (w, fw), (x, fx)
        v_new = torch.where(better, ww, torch.where((f_try <= fww) | (ww == xx), ww, torch.where((f_try <= fvv) | (vv == xx) | (vv == ww), u_try, vv)))
        fv_new = torch.where(better, fww, torch.where((f_try <= fww) | (ww == xx), fww, torch.where((f_try <= fvv) | (vv == xx) | (vv == ww), f_try, fvv)))
        w_new = torch.where(better, xx, torch.where((f_try <= fww) | (ww == xx), u_try, ww))
        fw_new = torch.where(better, fxx, torch.where((f_try <= fww) | (ww == xx), f_try, fww))
        x_new = torch.where(better, u_try, xx)
        fx_new = torch.where(better, f_try, fxx)

        a[idx] = a_new
        b[idx] = b_new
        v[idx] = v_new
        fv[idx] = fv_new
        w[idx] = w_new
        fw[idx] = fw_new
        x[idx] = x_new
        fx[idx] = fx_new
        active[idx] = (b_new - a_new) > (2.0 * u_tol)

    theta_cand = s0 * x / (1.0 - x)
    dev_cand, beta_cand, se_cand, valid_cand, _, _ = _run_pirls(
        theta_cand, all_idx, beta_ws, z_ws, max_iter=20, tol=1e-8
    )

    use_zero = valid_zero & ((~valid_cand) | (dev_zero <= dev_cand))
    theta_hat = torch.where(use_zero, theta_zero, theta_cand)
    beta_tx = torch.where(use_zero, beta_tx_zero, beta_cand)
    se_tx = torch.where(use_zero, se_tx_zero, se_cand)
    valid = valid_zero | valid_cand
    return theta_hat, beta_tx, se_tx, valid


# ---------------------------------------------------------------------------
# Public Entrypoints Called From R (via reticulate or JSON CLI)
# ---------------------------------------------------------------------------

def simulate_power_torch(
    outcome: str,
    trials: int,
    seed: int,
    I_clusters: int,
    Tp: int,
    K_vec: Any,
    delayed_tx_vec: Any,
    period_effects_vec: Any,
    adjust_period: bool = True,
    b1: float = 0.5,
    mu: float = 5.0,
    sdw: float = 2.0,
    sdc: float = 1.0,
    sdcp: float = 0.0,
    rho: float = 0.0,
    theta: float = 0.0,
    margin: float = 0.25,
    sig_level: float = 0.025,
    reml: bool = True,
    batch_size: Optional[int] = 10000,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run batched stepped-wedge NI power simulation on GPU/CPU with PyTorch."""
    if float(sdcp) > 0.0:
        import os as _os
        if _os.path.dirname(__file__) not in sys.path:
            sys.path.insert(0, _os.path.dirname(__file__))
        import swnisim_ar1_torch as _ar1

        k_first = float(K_vec[0]) if isinstance(K_vec, (list, tuple)) else float(K_vec)
        return _ar1.simulate_ar1_power_torch(
            outcome=outcome,
            trials=trials,
            seed=seed,
            I_clusters=I_clusters,
            Tp=Tp,
            K_scalar=k_first,
            delayed_tx_vec=delayed_tx_vec,
            period_effects_vec=period_effects_vec,
            adjust_period=adjust_period,
            b1=b1,
            mu=mu,
            sdw=sdw,
            sdc=sdc,
            sdcp=float(sdcp),
            rho=float(rho),
            theta=theta,
            margin=margin,
            sig_level=sig_level,
            reml=reml,
            batch_size=batch_size,
            device=device,
            resolve_device_fn=_resolve_device,
            make_gen_fn=_make_generator,
            randn_fn=_randn,
            binom_fn=_sample_binomial,
        )
    with torch.no_grad():
        dev, dtype, dev_label = _resolve_device(device)
        I_c = int(I_clusters)
        Tp_i = int(Tp)
        n_trials = int(trials)
        b_size = max(1, min(int(batch_size), n_trials))

        K_it = torch.tensor(K_vec, device=dev, dtype=dtype).reshape(I_c, Tp_i)
        tx_it = torch.tensor(delayed_tx_vec, device=dev, dtype=dtype).reshape(I_c, Tp_i)
        pe_t = torch.tensor(period_effects_vec, device=dev, dtype=dtype).reshape(1, Tp_i)

        # Build design tensor X_itp [I, Tp, p]
        p_dim = (Tp_i + 1) if adjust_period else 2
        X_itp = torch.zeros((I_c, Tp_i, p_dim), device=dev, dtype=dtype)
        X_itp[:, :, 0] = 1.0
        if adjust_period and Tp_i > 1:
            idx_p = torch.arange(1, Tp_i, device=dev)
            X_itp[:, idx_p, idx_p] = 1.0
        X_itp[:, :, -1] = tx_it

        # Check structural rank of X upfront (matches lme4 dropping collinear delayed_tx)
        x2d = X_itp.reshape(I_c * Tp_i, p_dim)
        _, info_x = torch.linalg.cholesky_ex(x2d.T @ x2d)
        if int(info_x.item()) != 0:
            return {
                "n_trials": n_trials,
                "n_rejections": 0,
                "n_failed": n_trials,
                "device": dev_label,
                "dtype": str(dtype).replace("torch.", ""),
            }

        gen = _make_generator(dev, int(seed))
        n_rejections = 0
        n_failed = 0

        if outcome == "continuous":
            precomp = _prepare_continuous_design(X_itp, K_it, float(sdw), float(sdc), bool(reml))
            if precomp is None:
                return {
                    "n_trials": n_trials,
                    "n_rejections": 0,
                    "n_failed": n_trials,
                    "device": dev_label,
                    "dtype": str(dtype).replace("torch.", ""),
                }
            eta_base = float(mu) + pe_t + float(theta) * tx_it  # [I, Tp]
            cell_sd = float(sdw) / torch.sqrt(K_it)             # [I, Tp]

            done = 0
            while done < n_trials:
                cur_B = min(b_size, n_trials - done)
                alpha = _randn((cur_B, I_c, 1), gen, dev, dtype) * float(sdc)
                eps = _randn((cur_B, I_c, Tp_i), gen, dev, dtype) * cell_sd.unsqueeze(0)
                y_bit = eta_base.unsqueeze(0) + alpha + eps

                _, beta_tx, se_tx, valid = _fit_continuous_from_y(
                    y_bit, X_itp, K_it, precomp, bool(reml)
                )
                z_ni = (beta_tx + float(margin)) / se_tx
                p_val = 0.5 * torch.erfc(z_ni * _INV_SQRT2)
                rej = valid & (p_val < float(sig_level))

                n_rejections += int(rej.sum().item())
                n_failed += int((~valid).sum().item())
                done += cur_B
        elif outcome == "binary":
            cuda_ext = _get_cuda_ext_safe() if dev.type == "cuda" else None
            eta_base = float(b1) + pe_t + float(theta) * tx_it  # [I, Tp]
            done = 0
            while done < n_trials:
                cur_B = min(b_size, n_trials - done)
                alpha = _randn((cur_B, I_c, 1), gen, dev, dtype) * float(sdc)
                probs = torch.sigmoid(eta_base.unsqueeze(0) + alpha)
                K_expand = K_it.unsqueeze(0).expand(cur_B, I_c, Tp_i)
                y_bit = _sample_binomial(K_expand, probs, gen, dev)

                _, beta_tx, se_tx, valid = _fit_binary_from_y(
                    y_bit, K_it, tx_it, bool(adjust_period), float(sdc), cuda_ext=cuda_ext
                )
                z_ni = (beta_tx + float(margin)) / se_tx
                p_val = 0.5 * torch.erfc(z_ni * _INV_SQRT2)
                rej = valid & (p_val < float(sig_level))

                n_rejections += int(rej.sum().item())
                n_failed += int((~valid).sum().item())
                done += cur_B
        else:
            raise ValueError(f"Unsupported outcome: {outcome}")

        has_k = bool(outcome == "binary" and dev.type == "cuda" and (2 <= Tp_i <= 8) and (_get_cuda_ext_safe() is not None))
        return {
            "n_trials": n_trials,
            "n_rejections": n_rejections,
            "n_failed": n_failed,
            "device": dev_label,
            "dtype": str(dtype).replace("torch.", ""),
            "cuda_kernel": has_k,
        }


def fit_datasets_torch(
    outcome: str,
    y_batch: Any,
    I_clusters: int,
    Tp: int,
    K_vec: Any,
    delayed_tx_vec: Any,
    adjust_period: bool = True,
    sdw: float = 2.0,
    sdc: float = 1.0,
    margin: float = 0.25,
    sig_level: float = 0.025,
    reml: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit pre-generated datasets [B, I * Tp] for deterministic validation against lme4."""
    with torch.no_grad():
        dev, dtype, dev_label = _resolve_device(device)
        I_c = int(I_clusters)
        Tp_i = int(Tp)
        K_it = torch.tensor(K_vec, device=dev, dtype=dtype).reshape(I_c, Tp_i)
        tx_it = torch.tensor(delayed_tx_vec, device=dev, dtype=dtype).reshape(I_c, Tp_i)
        y_t = torch.tensor(y_batch, device=dev, dtype=dtype).reshape(-1, I_c, Tp_i)
        B = y_t.shape[0]

        p_dim = (Tp_i + 1) if adjust_period else 2
        X_itp = torch.zeros((I_c, Tp_i, p_dim), device=dev, dtype=dtype)
        X_itp[:, :, 0] = 1.0
        if adjust_period and Tp_i > 1:
            idx_p = torch.arange(1, Tp_i, device=dev)
            X_itp[:, idx_p, idx_p] = 1.0
        X_itp[:, :, -1] = tx_it

        if outcome == "continuous":
            precomp = _prepare_continuous_design(X_itp, K_it, float(sdw), float(sdc), bool(reml))
            if precomp is None:
                return {
                    "theta": [float("nan")] * B,
                    "beta_tx": [float("nan")] * B,
                    "se_tx": [float("nan")] * B,
                    "reject": [None] * B,
                    "device": dev_label,
                }
            th_hat, beta_tx, se_tx, valid = _fit_continuous_from_y(
                y_t, X_itp, K_it, precomp, bool(reml)
            )
        else:
            cuda_ext = _get_cuda_ext_safe() if dev.type == "cuda" else None
            th_hat, beta_tx, se_tx, valid = _fit_binary_from_y(
                y_t, K_it, tx_it, bool(adjust_period), float(sdc), cuda_ext=cuda_ext
            )

        z_ni = (beta_tx + float(margin)) / se_tx
        p_val = 0.5 * torch.erfc(z_ni * _INV_SQRT2)
        rej = (p_val < float(sig_level)).to(torch.int64)

        valid_cpu = valid.to("cpu").tolist()
        th_cpu = th_hat.to("cpu").tolist()
        b_cpu = beta_tx.to("cpu").tolist()
        se_cpu = se_tx.to("cpu").tolist()
        rej_cpu = rej.to("cpu").tolist()

        return {
            "theta": [th_cpu[i] if valid_cpu[i] else None for i in range(B)],
            "beta_tx": [b_cpu[i] if valid_cpu[i] else None for i in range(B)],
            "se_tx": [se_cpu[i] if valid_cpu[i] else None for i in range(B)],
            "reject": [int(rej_cpu[i]) if valid_cpu[i] else None for i in range(B)],
            "device": dev_label,
        }


def simulate_grid_torch(scenarios: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Run a list of scenarios in one warm PyTorch session."""
    return [simulate_power_torch(**scen) for scen in scenarios]


def _cli_main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--probe":
        print(json.dumps(probe_torch()))
        return 0
    if len(argv) >= 4 and argv[1] == "--run":
        req_path, resp_path = argv[2], argv[3]
        with open(req_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        fn_name = payload.pop("_fn", "simulate_power_torch")
        if fn_name == "simulate_power_torch":
            res = simulate_power_torch(**payload)
        elif fn_name == "simulate_grid_torch":
            res = simulate_grid_torch(**payload)
        elif fn_name == "fit_datasets_torch":
            res = fit_datasets_torch(**payload)
        else:
            raise ValueError(f"Unknown CLI function: {fn_name}")
        with open(resp_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        return 0
    raise SystemExit("Usage: swnisim_torch.py [--probe | --run <in.json> <out.json>]")


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv))
