"""Batched PyTorch + Fused CUDA Kernel AR(1) mixed-model engine for swNIsim v0.3.0.

Replicates glmmTMB::glmmTMB fits for stepped-wedge non-inferiority trials with
both a cluster random intercept and a stationary AR(1) cluster-by-period random
effect:
  1. Continuous (REML = TRUE/FALSE, dispformula = ~ 1 + offset(-logK)):
       y ~ period + delayed_tx + (1 | cluster) + ar1(period + 0 | cluster)
  2. Binary (Laplace approximation, REML = FALSE, family = binomial()):
       cbind(success, fail) ~ period + delayed_tx + (1 | cluster) + ar1(period + 0 | cluster)

Tier 1 + Tier 2 accelerations in v0.3.0:
  - Fused CUDA kernels (`laplace_ar1_eval_cuda` and `cont_ar1_eval_cuda`) keep
    T x T Cholesky factorizations, inner Newton iterations, matrix inverses,
    and adjoint reductions entirely inside GPU registers & shared memory.
  - Safe accepted-step warm-starting of inner Laplace mode `u_ws` (never
    corrupted by rejected line-search trials).
  - Branchless GPU BFGS updates eliminating repeated CPU-GPU `.item()` syncs.
  - Guarded mixed-precision (`FP32` BFGS search -> `FP64` BFGS polish + `FP64`
    Hessian) for Binary AR(1) on CUDA GPUs, especially unlocking 1:64 FP64
    Ada/Turing GPUs (L40S, L4, T4) while retaining exact float64 MLE & pdHess.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import torch

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _get_cuda_ext_safe() -> Optional[Any]:
    try:
        pkg_dir = os.path.dirname(__file__)
        if pkg_dir and pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        import swnisim_cuda
        return swnisim_cuda.get_cuda_ext()
    except Exception:
        return None


def _use_mixed_binary_ar1(dev: torch.device) -> bool:
    if dev.type != "cuda":
        return False
    mode = os.environ.get("SWNISIM_AR1_PRECISION", "auto").strip().lower()
    if mode == "fp64":
        return False
    if mode == "mixed":
        return True
    # "auto": enable guarded FP32->FP64 polish on CUDA (huge win on L40S/L4/T4,
    # and also accelerates A100/H100 while preserving exact float64 MLE/pdHess).
    return True


def _batched_bfgs(
    fg_sub_fn: Callable[..., Tuple[torch.Tensor, ...]],
    x0: torch.Tensor,
    H0_inv: Optional[torch.Tensor] = None,
    u_ws: Optional[torch.Tensor] = None,
    max_iter: int = 120,
    gtol: float = 1e-6,
    rel_ftol: float = 1e-10,
    x_bound: float = 18.0,
    near_gtol: float = 1e-3,
    return_H_inv: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Batched BFGS with active-set compaction, branchless updates, and safe u_ws warm-start."""
    B, D = x0.shape
    dev = x0.device
    dtype = x0.dtype
    has_u = u_ws is not None

    x = x0.clone()
    all_idx = torch.arange(B, device=dev, dtype=torch.int64)
    if has_u:
        f, g, u0 = fg_sub_fn(x, all_idx, u_ws)
        finite_u0 = torch.isfinite(f) & torch.isfinite(u0).all(dim=(1, 2))
        u_ws[:] = torch.where(finite_u0.view(B, 1, 1), u0, u_ws)
    else:
        f, g = fg_sub_fn(x, all_idx)

    eye_D = torch.eye(D, device=dev, dtype=dtype)
    eye_D_1 = eye_D.unsqueeze(0)
    if H0_inv is not None:
        H_inv = H0_inv.clone()
        scaled_once = torch.ones(B, device=dev, dtype=torch.bool)
    else:
        H_inv = eye_D_1.expand(B, D, D).clone()
        scaled_once = torch.zeros(B, device=dev, dtype=torch.bool)

    finite0 = torch.isfinite(f) & torch.isfinite(g).all(dim=1)
    grad_done0 = finite0 & (torch.abs(g).amax(dim=1) < gtol)
    converged = grad_done0.clone()
    active = finite0 & (~grad_done0)

    eps_curv = 1e-12 if dtype == torch.float64 else 1e-6
    eps_tiny = 1e-30 if dtype == torch.float64 else 1e-18
    x_step_tol = 1.5e-8 if dtype == torch.float64 else 1e-5

    for _ in range(max_iter):
        idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        M = idx.numel()
        if M == 0:
            break

        x_a = x[idx]
        f_a = f[idx]
        g_a = g[idx]
        H_a = H_inv[idx]

        p_dir = -torch.bmm(H_a, g_a.unsqueeze(-1)).squeeze(-1)
        slope = (p_dir * g_a).sum(dim=1)
        bad_dir = (~torch.isfinite(slope)) | (slope >= 0.0)
        H_a = torch.where(bad_dir.view(M, 1, 1), eye_D_1, H_a)
        p_dir = torch.where(bad_dir.unsqueeze(1), -g_a, p_dir)
        slope = (p_dir * g_a).sum(dim=1)

        p_max = torch.abs(p_dir).amax(dim=1)
        step_cap = torch.where(
            p_max > 2.5, 2.5 / torch.clamp(p_max, min=1e-12), torch.ones_like(p_max)
        )
        p_dir = p_dir * step_cap.unsqueeze(1)
        slope = slope * step_cap

        alpha = torch.ones(M, device=dev, dtype=dtype)
        accepted = torch.zeros(M, device=dev, dtype=torch.bool)
        x_next = x_a.clone()
        f_next = f_a.clone()
        g_next = g_a.clone()
        u_a_base = u_ws[idx] if has_u else None
        u_next = u_a_base.clone() if has_u else None

        for _ls in range(14):
            sub_local = torch.nonzero(~accepted, as_tuple=False).reshape(-1)
            if sub_local.numel() == 0:
                break
            sub_global = idx[sub_local]
            x_try = x_a[sub_local] + alpha[sub_local].unsqueeze(1) * p_dir[sub_local]
            if has_u:
                f_try, g_try, u_try = fg_sub_fn(x_try, sub_global, u_a_base[sub_local])
            else:
                f_try, g_try = fg_sub_fn(x_try, sub_global)
            ok_ls = (
                torch.isfinite(f_try)
                & torch.isfinite(g_try).all(dim=1)
                & (f_try <= f_a[sub_local] + 1e-4 * alpha[sub_local] * slope[sub_local])
            )
            # Branchless scatter back into sub_local slots (no .item() sync!)
            x_next[sub_local] = torch.where(ok_ls.unsqueeze(1), x_try, x_next[sub_local])
            f_next[sub_local] = torch.where(ok_ls, f_try, f_next[sub_local])
            g_next[sub_local] = torch.where(ok_ls.unsqueeze(1), g_try, g_next[sub_local])
            if has_u:
                ok_u = ok_ls & torch.isfinite(u_try).all(dim=(1, 2))
                u_next[sub_local] = torch.where(ok_u.view(-1, 1, 1), u_try, u_next[sub_local])
            accepted[sub_local] = accepted[sub_local] | ok_ls
            alpha = torch.where(accepted, alpha, 0.5 * alpha)

        s = x_next - x_a
        y_d = g_next - g_a
        sy = (s * y_d).sum(dim=1)
        s_norm2 = (s * s).sum(dim=1)
        y_norm2 = (y_d * y_d).sum(dim=1)
        good_curv = (
            accepted
            & (sy > eps_curv * torch.sqrt(torch.clamp(s_norm2 * y_norm2, min=eps_tiny)))
        )

        # Nocedal & Wright Eq. 6.20 initial scaling (branchless)
        need_init_scale = good_curv & (~scaled_once[idx])
        gam0 = torch.clamp(sy / torch.clamp(y_norm2, min=eps_tiny), min=1e-4, max=1e4)
        H_a = torch.where(need_init_scale.view(M, 1, 1), gam0.view(M, 1, 1) * eye_D_1, H_a)
        scaled_once[idx] = scaled_once[idx] | need_init_scale

        rho_b = 1.0 / torch.clamp(sy, min=eps_tiny)
        Hy = torch.bmm(H_a, y_d.unsqueeze(-1)).squeeze(-1)
        yHy = (y_d * Hy).sum(dim=1)
        c1 = rho_b * (1.0 + rho_b * yHy)
        ssT = s.unsqueeze(2) * s.unsqueeze(1)
        HysT = Hy.unsqueeze(2) * s.unsqueeze(1)
        H_upd = H_a + c1.view(M, 1, 1) * ssT - rho_b.view(M, 1, 1) * (HysT + HysT.transpose(1, 2))
        H_a = torch.where(good_curv.view(M, 1, 1), H_upd, H_a)

        # If line search failed on non-Identity H_a, reset H_a to Identity once
        reset_mask = (~accepted) & (torch.abs(H_a - eye_D_1).amax(dim=(1, 2)) > 1e-7)
        H_a = torch.where(reset_mask.view(M, 1, 1), eye_D_1, H_a)

        g_inf = torch.abs(g_next).amax(dim=1)
        g_done = accepted & (g_inf < gtol)
        f_done = (
            accepted
            & (torch.abs(f_a - f_next) <= rel_ftol * (1.0 + torch.abs(f_a)))
            & (g_inf < near_gtol)
        )
        x_done = accepted & (torch.abs(s).amax(dim=1) < x_step_tol) & (g_inf < near_gtol)
        done_now = g_done | f_done | x_done

        out_of_bounds = torch.abs(x_next).amax(dim=1) > x_bound
        give_up = ((~accepted) & (~reset_mask)) | out_of_bounds

        x[idx] = x_next
        f[idx] = f_next
        g[idx] = g_next
        H_inv[idx] = H_a
        if has_u:
            u_ws[idx] = u_next
        converged[idx] = converged[idx] | (done_now & (~out_of_bounds))
        active[idx] = ~(done_now | give_up)

    near_conv = (
        torch.isfinite(f)
        & (torch.abs(g).amax(dim=1) < near_gtol)
        & (torch.abs(x).amax(dim=1) <= x_bound)
    )
    converged = converged | near_conv
    if return_H_inv:
        return x, f, g, converged, H_inv
    return x, f, g, converged


def _draw_ar1_batch(
    B: int,
    I_c: int,
    Tp: int,
    sdcp: float,
    rho: float,
    randn_fn: Callable[[Tuple[int, ...]], torch.Tensor],
    dev: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cp = torch.empty((B, I_c, Tp), device=dev, dtype=dtype)
    cp[:, :, 0] = randn_fn((B, I_c)) * float(sdcp)
    step_sd = float(sdcp) * math.sqrt(max(0.0, 1.0 - float(rho) ** 2))
    rho_f = float(rho)
    for t in range(1, Tp):
        cp[:, :, t] = rho_f * cp[:, :, t - 1] + randn_fn((B, I_c)) * step_sd
    return cp


# ---------------------------------------------------------------------------
# 1. Batched Continuous AR(1) Engine (matches glmmTMB Gaussian + ar1 + REML)
# ---------------------------------------------------------------------------

def _fit_continuous_ar1_batch(
    Y_bit: torch.Tensor,          # [B, I, Tp]
    X_itp: torch.Tensor,          # [I, Tp, p]
    M_XX: torch.Tensor,           # [Tp, Tp, p, p]
    K_t: torch.Tensor,            # [Tp]
    reml: bool,
    cuda_ext: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit continuous LMM with (1|cluster) + ar1(period+0|cluster). Returns (beta_tx, se_tx, valid)."""
    B, I_c, Tp = Y_bit.shape
    p_dim = X_itp.shape[2]
    dev = Y_bit.device
    dtype = Y_bit.dtype
    tiny = torch.finfo(dtype).tiny

    # Sufficient statistics per trial: S_YY [B, Tp, Tp] and G_YX [B, Tp, Tp, p]
    S_YY = torch.einsum("bit,bis->bts", Y_bit, Y_bit).contiguous()
    G_YX = torch.einsum("itp,bis->btsp", X_itp, Y_bit).contiguous()
    M_XX_c = M_XX.contiguous()
    K_t_c = K_t.contiguous()

    use_cuda_kernel = (
        cuda_ext is not None
        and dev.type == "cuda"
        and 2 <= Tp <= 8
        and (p_dim == 2 or p_dim == Tp + 1)
    )

    if use_cuda_kernel:
        def _eval_sub(
            psi_sub: torch.Tensor, idx: torch.Tensor, need_aux: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            f_val, grad, beta, ainv_last = cuda_ext.cont_ar1_eval_cuda(
                psi_sub, idx, S_YY, G_YX, M_XX_c, K_t_c, int(I_c), bool(reml), bool(need_aux)
            )
            if need_aux:
                return f_val, grad, beta, ainv_last
            return f_val, grad, None, None
    else:
        t_idx = torch.arange(Tp, device=dev, dtype=dtype)
        dist = torch.abs(t_idx.unsqueeze(1) - t_idx.unsqueeze(0))
        dist_m1 = torch.clamp(dist - 1.0, min=0.0)
        off_diag_mask = (dist > 0.0).to(dtype)

        eye_Tp = torch.eye(Tp, device=dev, dtype=dtype)
        eye_p = torch.eye(p_dim, device=dev, dtype=dtype)
        diag_t = torch.arange(Tp, device=dev)

        def _eval_sub(
            psi_sub: torch.Tensor, idx: torch.Tensor, need_aux: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            M = psi_sub.shape[0]
            p0 = torch.clamp(psi_sub[:, 0], min=-35.0, max=35.0)
            p1 = torch.clamp(psi_sub[:, 1], min=-18.0, max=18.0)
            p2 = torch.clamp(psi_sub[:, 2], min=-18.0, max=18.0)
            p3 = psi_sub[:, 3]

            ve = torch.exp(p0).unsqueeze(1) / K_t_c.unsqueeze(0)
            vc = torch.exp(2.0 * p1)
            vcp = torch.exp(2.0 * p2)
            inv_s = 1.0 / torch.sqrt(1.0 + p3 * p3)
            rho = p3 * inv_s

            R = rho.view(M, 1, 1) ** dist.unsqueeze(0)
            Sig = vc.view(M, 1, 1) + vcp.view(M, 1, 1) * R
            Sig[:, diag_t, diag_t] = Sig[:, diag_t, diag_t] + ve

            L_Sig, info_s = torch.linalg.cholesky_ex(Sig)
            bad_s = info_s != 0
            L_Sig = torch.where(bad_s.view(M, 1, 1), eye_Tp.unsqueeze(0), L_Sig)
            Sinv = torch.cholesky_solve(eye_Tp.unsqueeze(0).expand(M, Tp, Tp), L_Sig)

            A = torch.einsum("mts,tspq->mpq", Sinv, M_XX_c)
            L_A, info_a = torch.linalg.cholesky_ex(A)
            bad = bad_s | (info_a != 0)
            L_A = torch.where(bad.view(M, 1, 1), eye_p.unsqueeze(0), L_A)
            Ainv = torch.cholesky_solve(eye_p.unsqueeze(0).expand(M, p_dim, p_dim), L_A)

            G_sub = G_YX[idx]
            SYY_sub = S_YY[idx]
            rhs = torch.einsum("mts,mtsp->mp", Sinv, G_sub)
            beta = torch.einsum("mpq,mq->mp", Ainv, rhs)

            XY_b = torch.einsum("mtsp,mp->mts", G_sub, beta)
            C_mat = torch.einsum("mp,mq->mpq", beta, beta)
            if reml:
                C_mat = C_mat + Ainv
            XCX = torch.einsum("tspq,mpq->mts", M_XX_c, C_mat)
            mid = SYY_sub - XY_b - XY_b.transpose(1, 2) + XCX

            quad = (Sinv * SYY_sub).sum(dim=(1, 2)) - (beta * rhs).sum(dim=1)
            logdet_Sig = 2.0 * torch.log(torch.clamp(torch.diagonal(L_Sig, dim1=-2, dim2=-1), min=tiny)).sum(dim=1)
            if reml:
                logdet_A = 2.0 * torch.log(torch.clamp(torch.diagonal(L_A, dim1=-2, dim2=-1), min=tiny)).sum(dim=1)
                f_val = 0.5 * (float(I_c) * logdet_Sig + logdet_A + quad)
            else:
                f_val = 0.5 * (float(I_c) * logdet_Sig + quad)

            Sinv_mid_Sinv = Sinv @ mid @ Sinv
            G_Sig = 0.5 * (float(I_c) * Sinv - Sinv_mid_Sinv)

            g0 = (torch.diagonal(G_Sig, dim1=-2, dim2=-1) * ve).sum(dim=1)
            g1 = 2.0 * vc * G_Sig.sum(dim=(1, 2))
            g2 = 2.0 * vcp * (G_Sig * R).sum(dim=(1, 2))
            dR = dist.unsqueeze(0) * (rho.view(M, 1, 1) ** dist_m1.unsqueeze(0)) * off_diag_mask.unsqueeze(0)
            g3 = vcp * (inv_s ** 3) * (G_Sig * dR).sum(dim=(1, 2))
            grad = torch.stack([g0, g1, g2, g3], dim=1)

            f_val = torch.where(bad, torch.full_like(f_val, float("inf")), f_val)
            grad = torch.where(bad.unsqueeze(1), torch.zeros_like(grad), grad)
            if need_aux:
                return f_val, grad, beta, Ainv[:, -1, -1]
            return f_val, grad, None, None

    x0 = torch.zeros((B, 4), device=dev, dtype=dtype)
    psi_hat, f_hat, _, conv = _batched_bfgs(
        lambda x_s, idx_s: _eval_sub(x_s, idx_s, False)[:2],
        x0,
        max_iter=90,
        gtol=1e-6,
        rel_ftol=1e-10,
        x_bound=16.0,
    )

    all_idx = torch.arange(B, device=dev, dtype=torch.int64)
    _, _, beta_hat, ainv_last_hat = _eval_sub(psi_hat, all_idx, True)
    assert beta_hat is not None and ainv_last_hat is not None

    # Central-difference Hessian of analytic gradient (matches TMB::sdreport optimHess)
    eps = 1e-4
    H_psi = torch.empty((B, 4, 4), device=dev, dtype=dtype)
    J_beta = torch.empty((B, p_dim, 4), device=dev, dtype=dtype)
    for j in range(4):
        ej = torch.zeros((1, 4), device=dev, dtype=dtype)
        ej[0, j] = eps
        _, gp, bp, _ = _eval_sub(psi_hat + ej, all_idx, True)
        _, gm, bm, _ = _eval_sub(psi_hat - ej, all_idx, True)
        assert bp is not None and bm is not None
        H_psi[:, :, j] = (gp - gm) / (2.0 * eps)
        J_beta[:, :, j] = (bp - bm) / (2.0 * eps)
    H_psi = 0.5 * (H_psi + H_psi.transpose(1, 2))

    L_H, info_h = torch.linalg.cholesky_ex(H_psi)
    pd_hess = (info_h == 0) & torch.isfinite(H_psi).all(dim=(1, 2))
    diag_LH = torch.diagonal(L_H, dim1=-2, dim2=-1)
    pd_hess = pd_hess & (diag_LH.amin(dim=1) > 1e-5)

    L_H_safe = torch.where(
        pd_hess.view(B, 1, 1),
        L_H,
        torch.eye(4, device=dev, dtype=dtype).unsqueeze(0),
    )
    j_tx = J_beta[:, -1, :]
    Hinv_jtx = torch.cholesky_solve(j_tx.unsqueeze(-1), L_H_safe).squeeze(-1)
    var_corr = torch.clamp((j_tx * Hinv_jtx).sum(dim=1), min=0.0)
    var_tx = ainv_last_hat + var_corr
    se_tx = torch.sqrt(torch.clamp(var_tx, min=tiny))
    beta_tx = beta_hat[:, -1]

    valid = (
        conv
        & pd_hess
        & torch.isfinite(f_hat)
        & torch.isfinite(beta_tx)
        & torch.isfinite(se_tx)
        & (se_tx > 0.0)
    )
    return beta_tx, se_tx, valid


# ---------------------------------------------------------------------------
# 2. Batched Binary AR(1) Engine (matches glmmTMB Binomial + ar1 + Laplace)
# ---------------------------------------------------------------------------

def _make_laplace_evaluator(
    Y_bit: torch.Tensor,          # [B, I, Tp]
    X_itp: torch.Tensor,          # [I, Tp, p]
    K_it: torch.Tensor,           # [I, Tp]
    cuda_ext: Optional[Any],
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build either fused CUDA kernel or pure-PyTorch Laplace evaluator for given dtype."""
    B, I_c, Tp = Y_bit.shape
    p_dim = X_itp.shape[2]
    dev = Y_bit.device
    dtype = Y_bit.dtype
    tiny = torch.finfo(dtype).tiny

    Y_c = Y_bit.contiguous()
    X_c = X_itp.contiguous()
    K_c = K_it.contiguous()

    use_cuda_kernel = (
        cuda_ext is not None
        and dev.type == "cuda"
        and 2 <= Tp <= 8
        and p_dim <= 9
    )

    if use_cuda_kernel:
        def _eval_cuda(
            params_sub: torch.Tensor,
            idx: torch.Tensor,
            u_start: torch.Tensor,
            n_inner: int,
            inner_tol: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            f_val, grad, u_out = cuda_ext.laplace_ar1_eval_cuda(
                params_sub, idx, Y_c, X_c, K_c, u_start, int(n_inner), float(inner_tol)
            )
            return f_val, grad, u_out

        return _eval_cuda

    t_idx = torch.arange(Tp, device=dev, dtype=dtype)
    dist = torch.abs(t_idx.unsqueeze(1) - t_idx.unsqueeze(0))
    dist_m1 = torch.clamp(dist - 1.0, min=0.0)
    off_diag_mask = (dist > 0.0).to(dtype)
    eye_Tp = torch.eye(Tp, device=dev, dtype=dtype)
    diag_t = torch.arange(Tp, device=dev)
    K_b = K_c.unsqueeze(0)

    def _eval_torch(
        params_sub: torch.Tensor,
        idx: torch.Tensor,
        u_start: torch.Tensor,
        n_inner: int,
        inner_tol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = params_sub.shape[0]
        beta = params_sub[:, :p_dim]
        p1 = torch.clamp(params_sub[:, p_dim], min=-16.0, max=16.0)
        p2 = torch.clamp(params_sub[:, p_dim + 1], min=-16.0, max=16.0)
        p3 = params_sub[:, p_dim + 2]

        vc = torch.exp(2.0 * p1)
        vcp = torch.exp(2.0 * p2)
        inv_s = 1.0 / torch.sqrt(1.0 + p3 * p3)
        rho = p3 * inv_s

        R = rho.view(M, 1, 1) ** dist.unsqueeze(0)
        Om = vc.view(M, 1, 1) + vcp.view(M, 1, 1) * R
        Om[:, diag_t, diag_t] = Om[:, diag_t, diag_t] + 1e-13

        L_Om, info_om = torch.linalg.cholesky_ex(Om)
        bad_om = info_om != 0
        L_Om = torch.where(bad_om.view(M, 1, 1), eye_Tp.unsqueeze(0), L_Om)
        Om_inv = torch.cholesky_solve(eye_Tp.unsqueeze(0).expand(M, Tp, Tp), L_Om)
        logdet_Om = 2.0 * torch.log(torch.clamp(torch.diagonal(L_Om, dim1=-2, dim2=-1), min=tiny)).sum(dim=1)

        Y_sub = Y_c[idx]
        eta_f = torch.einsum("itp,mp->mit", X_c, beta)
        u = u_start.clone()

        for it in range(n_inner):
            eta = torch.clamp(eta_f + u, min=-28.0, max=28.0)
            mu_p = torch.sigmoid(eta)
            W = torch.clamp(K_b * mu_p * (1.0 - mu_p), min=1e-15)
            u_Ominv = torch.einsum("mit,mts->mis", u, Om_inv)
            g_u = (Y_sub - K_b * mu_p) - u_Ominv

            H_u = Om_inv.unsqueeze(1).expand(M, I_c, Tp, Tp).clone()
            H_u[:, :, diag_t, diag_t] = H_u[:, :, diag_t, diag_t] + W

            L_Hu, info_hu = torch.linalg.cholesky_ex(H_u)
            L_Hu = torch.where(
                (info_hu != 0).unsqueeze(-1).unsqueeze(-1),
                eye_Tp.view(1, 1, Tp, Tp),
                L_Hu,
            )
            du = torch.cholesky_solve(g_u.unsqueeze(-1), L_Hu).squeeze(-1)
            u = u + du
            # Check convergence every 2 iterations (after iter 1) to reduce GPU-CPU syncs
            if it >= 1 and (it % 2 == 1) and float(torch.abs(du).amax().item()) < inner_tol:
                break

        eta = torch.clamp(eta_f + u, min=-28.0, max=28.0)
        mu_p = torch.sigmoid(eta)
        W = torch.clamp(K_b * mu_p * (1.0 - mu_p), min=1e-15)

        H_u = Om_inv.unsqueeze(1).expand(M, I_c, Tp, Tp).clone()
        H_u[:, :, diag_t, diag_t] = H_u[:, :, diag_t, diag_t] + W
        L_Hu, info_hu = torch.linalg.cholesky_ex(H_u)
        bad = bad_om | (info_hu != 0).any(dim=1)
        L_Hu = torch.where(
            (info_hu != 0).unsqueeze(-1).unsqueeze(-1),
            eye_Tp.view(1, 1, Tp, Tp),
            L_Hu,
        )

        eye_expand = eye_Tp.view(1, 1, Tp, Tp).expand(M, I_c, Tp, Tp)
        Hu_inv = torch.cholesky_solve(eye_expand, L_Hu)
        h_diag = torch.diagonal(Hu_inv, dim1=-2, dim2=-1)
        logdet_Hu = 2.0 * torch.log(
            torch.clamp(torch.diagonal(L_Hu, dim1=-2, dim2=-1), min=tiny)
        ).sum(dim=(1, 2))

        log1p_e = torch.clamp(eta, min=0.0) + torch.log1p(torch.exp(-torch.abs(eta)))
        ll_y = (Y_sub * eta - K_b * log1p_e).sum(dim=(1, 2))
        quad_u = 0.5 * torch.einsum("mit,mts,mis->m", u, Om_inv, u)
        f_val = -ll_y + quad_u + 0.5 * float(I_c) * logdet_Om + 0.5 * logdet_Hu

        v = 0.5 * h_diag * W * (1.0 - 2.0 * mu_p)
        lam = torch.einsum("mits,mis->mit", Hu_inv, v)
        grad_eta_f = -(Y_sub - K_b * mu_p) + v - W * lam
        grad_beta = torch.einsum("itp,mit->mp", X_c, grad_eta_f)

        uTu = torch.einsum("mit,mis->mts", u, u)
        lamTu = torch.einsum("mit,mis->mts", lam, u)
        G_Ominv = 0.5 * (uTu - float(I_c) * Om + Hu_inv.sum(dim=1)) - 0.5 * (
            lamTu + lamTu.transpose(1, 2)
        )
        G_Om = -(Om_inv @ G_Ominv @ Om_inv)

        gp1 = 2.0 * vc * G_Om.sum(dim=(1, 2))
        gp2 = 2.0 * vcp * (G_Om * R).sum(dim=(1, 2))
        dR = dist.unsqueeze(0) * (rho.view(M, 1, 1) ** dist_m1.unsqueeze(0)) * off_diag_mask.unsqueeze(0)
        gp3 = vcp * (inv_s ** 3) * (G_Om * dR).sum(dim=(1, 2))
        grad = torch.cat([grad_beta, torch.stack([gp1, gp2, gp3], dim=1)], dim=1)

        f_val = torch.where(bad, torch.full_like(f_val, float("inf")), f_val)
        grad = torch.where(bad.unsqueeze(1), torch.zeros_like(grad), grad)
        return f_val, grad, u

    return _eval_torch


def _fit_binary_ar1_batch(
    Y_bit: torch.Tensor,          # [B, I, Tp]
    X_itp: torch.Tensor,          # [I, Tp, p]
    K_it: torch.Tensor,           # [I, Tp]
    cuda_ext: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit binary GLMM with (1|cluster) + ar1(period+0|cluster) via Laplace."""
    B, I_c, Tp = Y_bit.shape
    p_dim = X_itp.shape[2]
    D = p_dim + 3
    dev = Y_bit.device
    dtype = Y_bit.dtype
    tiny = torch.finfo(dtype).tiny
    eye_D = torch.eye(D, device=dev, dtype=dtype)
    K_b = K_it.unsqueeze(0)

    # Fast GLM IRLS initialization for beta at u = 0 (3 steps)
    tot_y = Y_bit.sum(dim=(1, 2))
    tot_K = float(K_it.sum().item())
    p_emp = (tot_y + 0.5) / (tot_K + 1.0)
    beta_init = torch.zeros((B, p_dim), device=dev, dtype=dtype)
    beta_init[:, 0] = torch.log(p_emp / (1.0 - p_emp))
    for _ in range(3):
        eta0 = torch.clamp(torch.einsum("itp,bp->bit", X_itp, beta_init), min=-15.0, max=15.0)
        mu0 = torch.sigmoid(eta0)
        W0 = torch.clamp(K_b * mu0 * (1.0 - mu0), min=1e-9)
        g0 = Y_bit - K_b * mu0
        XWX = torch.einsum("itp,bit,itq->bpq", X_itp, W0, X_itp)
        XWg = torch.einsum("itp,bit->bp", X_itp, g0)
        L0, info0 = torch.linalg.cholesky_ex(XWX)
        ok0 = info0 == 0
        L0_safe = torch.where(ok0.view(B, 1, 1), L0, torch.eye(p_dim, device=dev, dtype=dtype).unsqueeze(0))
        db0 = torch.cholesky_solve(XWg.unsqueeze(-1), L0_safe).squeeze(-1)
        beta_init = torch.where(ok0.unsqueeze(1), beta_init + db0, beta_init)

    # Precondition initial BFGS inverse Hessian using GLM Fisher inverse for beta
    H0_inv = eye_D.unsqueeze(0).expand(B, D, D).clone() * 0.25
    eye_p = torch.eye(p_dim, device=dev, dtype=dtype).unsqueeze(0).expand(B, p_dim, p_dim)
    eta0 = torch.clamp(torch.einsum("itp,bp->bit", X_itp, beta_init), min=-15.0, max=15.0)
    mu0 = torch.sigmoid(eta0)
    W0 = torch.clamp(K_b * mu0 * (1.0 - mu0), min=1e-9)
    XWX = torch.einsum("itp,bit,itq->bpq", X_itp, W0, X_itp)
    L0, info0 = torch.linalg.cholesky_ex(XWX)
    ok0 = info0 == 0
    L0_safe = torch.where(ok0.view(B, 1, 1), L0, torch.eye(p_dim, device=dev, dtype=dtype).unsqueeze(0))
    inv_XWX = torch.cholesky_solve(eye_p, L0_safe)
    H0_inv[:, :p_dim, :p_dim] = torch.where(ok0.view(B, 1, 1), inv_XWX, H0_inv[:, :p_dim, :p_dim])

    x0 = torch.zeros((B, D), device=dev, dtype=dtype)
    x0[:, :p_dim] = beta_init

    eval_fp64 = _make_laplace_evaluator(Y_bit, X_itp, K_it, cuda_ext)
    u_ws = torch.zeros((B, I_c, Tp), device=dev, dtype=dtype)
    x_start = x0
    H_start = H0_inv

    # Optional guarded FP32 search phase -> FP64 exact polish (huge speedup on L40S/L4/T4 and A100)
    if dtype == torch.float64 and _use_mixed_binary_ar1(dev):
        eval_fp32 = _make_laplace_evaluator(
            Y_bit.to(torch.float32),
            X_itp.to(torch.float32),
            K_it.to(torch.float32),
            cuda_ext,
        )
        u_ws32 = torch.zeros((B, I_c, Tp), device=dev, dtype=torch.float32)
        x32, f32, _, _, H32 = _batched_bfgs(
            lambda x_s, idx_s, u_s: eval_fp32(x_s, idx_s, u_s, 10, 1e-5),
            x0.to(torch.float32),
            H0_inv=H0_inv.to(torch.float32),
            u_ws=u_ws32,
            max_iter=65,
            gtol=5e-2,
            rel_ftol=1e-6,
            x_bound=14.5,
            near_gtol=1e-1,
            return_H_inv=True,
        )
        ok32 = (
            torch.isfinite(f32)
            & torch.isfinite(x32).all(dim=1)
            & (torch.abs(x32).amax(dim=1) <= 14.0)
            & torch.isfinite(u_ws32).all(dim=(1, 2))
        )
        x_start = torch.where(ok32.unsqueeze(1), x32.to(dtype), x0)
        u_ws = torch.where(ok32.view(B, 1, 1), u_ws32.to(dtype), u_ws)
        ok_H32 = ok32 & torch.isfinite(H32).all(dim=(1, 2)) & (torch.abs(H32).amax(dim=(1, 2)) < 1e4)
        H_start = torch.where(ok_H32.view(B, 1, 1), H32.to(dtype), H0_inv)

    # Exact float64 BFGS (either full solve or fast 2-6 iter superlinear polish from FP32 warm start)
    x_hat, f_hat, _, conv = _batched_bfgs(
        lambda x_s, idx_s, u_s: eval_fp64(x_s, idx_s, u_s, 12, 1e-11),
        x_start,
        H0_inv=H_start,
        u_ws=u_ws,
        max_iter=150,
        gtol=1e-5,
        rel_ftol=1e-10,
        x_bound=15.0,
        near_gtol=1e-3,
    )

    # Polish u_star tightly at x_hat (starts from warm u_ws, usually 1-2 Newton steps)
    all_idx = torch.arange(B, device=dev, dtype=torch.int64)
    _, _, u_star = eval_fp64(x_hat, all_idx, u_ws, 10, 1e-12)
    u_star = torch.where(torch.isfinite(u_star), u_star, torch.zeros_like(u_star))

    # Central-difference Hessian around converged x_hat starting from u_star (2-4 Newton steps suffice)
    eps = 1e-4
    H_full = torch.empty((B, D, D), device=dev, dtype=dtype)
    for j in range(D):
        ej = torch.zeros((1, D), device=dev, dtype=dtype)
        ej[0, j] = eps
        _, gp, _ = eval_fp64(x_hat + ej, all_idx, u_star, 4, 1e-12)
        _, gm, _ = eval_fp64(x_hat - ej, all_idx, u_star, 4, 1e-12)
        H_full[:, :, j] = (gp - gm) / (2.0 * eps)
    H_full = 0.5 * (H_full + H_full.transpose(1, 2))

    L_H, info_h = torch.linalg.cholesky_ex(H_full)
    pd_hess = (info_h == 0) & torch.isfinite(H_full).all(dim=(1, 2))
    L_H_safe = torch.where(pd_hess.view(B, 1, 1), L_H, eye_D.unsqueeze(0))

    e_tx = torch.zeros((B, D, 1), device=dev, dtype=dtype)
    e_tx[:, p_dim - 1, 0] = 1.0
    col_tx = torch.cholesky_solve(e_tx, L_H_safe).squeeze(-1)
    var_tx = col_tx[:, p_dim - 1]
    se_tx = torch.sqrt(torch.clamp(var_tx, min=tiny))
    beta_tx = x_hat[:, p_dim - 1]

    valid = (
        conv
        & pd_hess
        & torch.isfinite(f_hat)
        & torch.isfinite(beta_tx)
        & torch.isfinite(se_tx)
        & (se_tx > 0.0)
    )
    return beta_tx, se_tx, valid


def simulate_ar1_power_torch(
    outcome: str,
    trials: int,
    seed: int,
    I_clusters: int,
    Tp: int,
    K_scalar: float,
    delayed_tx_vec: Any,
    period_effects_vec: Any,
    adjust_period: bool = True,
    b1: float = 0.5,
    mu: float = 5.0,
    sdw: float = 2.0,
    sdc: float = 1.0,
    sdcp: float = 0.5,
    rho: float = 0.1,
    theta: float = 0.0,
    margin: float = 0.25,
    sig_level: float = 0.025,
    reml: bool = True,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    resolve_device_fn: Any = None,
    make_gen_fn: Any = None,
    randn_fn: Any = None,
    binom_fn: Any = None,
) -> Dict[str, Any]:
    """Run batched AR(1) stepped-wedge NI power simulation on GPU/CPU."""
    with torch.no_grad():
        dev, dtype, dev_label = resolve_device_fn(device)
        I_c = int(I_clusters)
        Tp_i = int(Tp)
        n_trials = int(trials)

        cuda_ext = _get_cuda_ext_safe() if dev.type == "cuda" else None
        has_kernel = (cuda_ext is not None) and (2 <= Tp_i <= 8)

        if batch_size is None or int(batch_size) <= 0:
            if outcome == "continuous":
                b_size = min(n_trials, (15000 if has_kernel else 5000) if dev.type == "cuda" else 1000)
            else:
                b_size = min(n_trials, (10000 if has_kernel else 2500) if dev.type == "cuda" else 250)
        else:
            b_size = max(1, min(int(batch_size), n_trials))

        tx_it = torch.tensor(delayed_tx_vec, device=dev, dtype=dtype).reshape(I_c, Tp_i)
        pe_t = torch.tensor(period_effects_vec, device=dev, dtype=dtype).reshape(1, Tp_i)
        K_val = float(K_scalar)
        K_t = torch.full((Tp_i,), K_val, device=dev, dtype=dtype)
        K_it = torch.full((I_c, Tp_i), K_val, device=dev, dtype=dtype)

        p_dim = (Tp_i + 1) if adjust_period else 2
        X_itp = torch.zeros((I_c, Tp_i, p_dim), device=dev, dtype=dtype)
        X_itp[:, :, 0] = 1.0
        if adjust_period and Tp_i > 1:
            idx_p = torch.arange(1, Tp_i, device=dev)
            X_itp[:, idx_p, idx_p] = 1.0
        X_itp[:, :, -1] = tx_it

        gen = make_gen_fn(dev, int(seed))
        n_rejections = 0
        n_failed = 0

        if outcome == "continuous":
            M_XX = torch.einsum("itp,isq->tspq", X_itp, X_itp)
            eta_base = float(mu) + pe_t + float(theta) * tx_it
            cell_sd = float(sdw) / math.sqrt(K_val)

            done = 0
            while done < n_trials:
                cur_B = min(b_size, n_trials - done)
                alpha = randn_fn((cur_B, I_c, 1), gen, dev, dtype) * float(sdc)
                cp = _draw_ar1_batch(
                    cur_B, I_c, Tp_i, float(sdcp), float(rho),
                    lambda shp: randn_fn(shp, gen, dev, dtype), dev, dtype,
                )
                eps = randn_fn((cur_B, I_c, Tp_i), gen, dev, dtype) * cell_sd
                Y_bit = eta_base.unsqueeze(0) + alpha + cp + eps

                beta_tx, se_tx, valid = _fit_continuous_ar1_batch(
                    Y_bit, X_itp, M_XX, K_t, bool(reml), cuda_ext=cuda_ext
                )
                z_ni = (beta_tx + float(margin)) / se_tx
                p_val = 0.5 * torch.erfc(z_ni * _INV_SQRT2)
                rej = valid & (p_val < float(sig_level))

                n_rejections += int(rej.sum().item())
                n_failed += int((~valid).sum().item())
                done += cur_B
        elif outcome == "binary":
            eta_base = float(b1) + pe_t + float(theta) * tx_it
            done = 0
            while done < n_trials:
                cur_B = min(b_size, n_trials - done)
                alpha = randn_fn((cur_B, I_c, 1), gen, dev, dtype) * float(sdc)
                cp = _draw_ar1_batch(
                    cur_B, I_c, Tp_i, float(sdcp), float(rho),
                    lambda shp: randn_fn(shp, gen, dev, dtype), dev, dtype,
                )
                probs = torch.sigmoid(eta_base.unsqueeze(0) + alpha + cp)
                K_exp = K_it.unsqueeze(0).expand(cur_B, I_c, Tp_i)
                Y_bit = binom_fn(K_exp, probs, gen, dev)

                beta_tx, se_tx, valid = _fit_binary_ar1_batch(
                    Y_bit, X_itp, K_it, cuda_ext=cuda_ext
                )
                z_ni = (beta_tx + float(margin)) / se_tx
                p_val = 0.5 * torch.erfc(z_ni * _INV_SQRT2)
                rej = valid & (p_val < float(sig_level))

                n_rejections += int(rej.sum().item())
                n_failed += int((~valid).sum().item())
                done += cur_B
        else:
            raise ValueError(f"Unsupported outcome: {outcome}")

        return {
            "n_trials": n_trials,
            "n_rejections": n_rejections,
            "n_failed": n_failed,
            "device": dev_label,
            "dtype": str(dtype).replace("torch.", ""),
            "cuda_kernel": bool(has_kernel),
        }
