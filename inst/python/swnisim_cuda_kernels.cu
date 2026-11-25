#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int MAX_T = 8;
constexpr int MAX_P = 9;
constexpr int MAX_SYM_T = MAX_T * (MAX_T + 1) / 2;
constexpr int MAX_SYM_P = MAX_P * (MAX_P + 1) / 2;

__device__ __forceinline__ int sym_idx(int r, int c) {
    return (r * (r + 1)) / 2 + c;
}

template <typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

template <typename T>
__device__ __forceinline__ T tiny_val();

template <>
__device__ __forceinline__ double tiny_val<double>() {
    return 1e-300;
}

template <>
__device__ __forceinline__ float tiny_val<float>() {
    return 1e-35f;
}

template <typename T>
__device__ __forceinline__ T inf_val();

template <>
__device__ __forceinline__ double inf_val<double>() {
    return __longlong_as_double(0x7ff0000000000000LL);
}

template <>
__device__ __forceinline__ float inf_val<float>() {
    return __int_as_float(0x7f800000);
}

template <typename T>
__device__ __forceinline__ T sigmoid_val(T x) {
    if (x >= T(0)) {
        T e = exp(-x);
        return T(1) / (T(1) + e);
    } else {
        T e = exp(x);
        return e / (T(1) + e);
    }
}

template <typename T>
__device__ __forceinline__ T log1p_exp_val(T x) {
    T ax = fabs(x);
    return (x > T(0) ? x : T(0)) + log1p(exp(-ax));
}

// In-place Cholesky factorization of symmetric lower-triangle A (size N*(N+1)/2).
// Returns true on success (strictly positive finite diagonal), false otherwise.
template <typename scalar_t, int N>
__device__ __forceinline__ bool cholesky_lower_inplace(scalar_t* L) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            scalar_t sum = L[sym_idx(i, j)];
            for (int k = 0; k < j; ++k) {
                sum -= L[sym_idx(i, k)] * L[sym_idx(j, k)];
            }
            if (i == j) {
                if (!(sum > scalar_t(0)) || !isfinite(sum)) {
                    return false;
                }
                L[sym_idx(i, j)] = sqrt(sum);
            } else {
                L[sym_idx(i, j)] = sum / L[sym_idx(j, j)];
            }
        }
    }
    return true;
}

// Solve (L * L^T) * x = b where L is lower-triangular packed (N*(N+1)/2).
template <typename scalar_t, int N>
__device__ __forceinline__ void cholesky_solve_vec(
    const scalar_t* L,
    const scalar_t* b,
    scalar_t* x
) {
    for (int i = 0; i < N; ++i) {
        scalar_t s = b[i];
        for (int j = 0; j < i; ++j) {
            s -= L[sym_idx(i, j)] * x[j];
        }
        x[i] = s / L[sym_idx(i, i)];
    }
    for (int i = N - 1; i >= 0; --i) {
        scalar_t s = x[i];
        for (int j = i + 1; j < N; ++j) {
            s -= L[sym_idx(j, i)] * x[j];
        }
        x[i] = s / L[sym_idx(i, i)];
    }
}

// In-place inversion of lower-triangular packed matrix L -> L^{-1}.
template <typename scalar_t, int N>
__device__ __forceinline__ void invert_lower_inplace(scalar_t* L) {
    scalar_t row_tmp[N];
    for (int r = 0; r < N; ++r) {
        scalar_t inv_diag = scalar_t(1) / L[sym_idx(r, r)];
        for (int j = 0; j < r; ++j) {
            scalar_t s = scalar_t(0);
            for (int c = j; c < r; ++c) {
                s += L[sym_idx(r, c)] * L[sym_idx(c, j)];
            }
            row_tmp[j] = -s * inv_diag;
        }
        for (int j = 0; j < r; ++j) {
            L[sym_idx(r, j)] = row_tmp[j];
        }
        L[sym_idx(r, r)] = inv_diag;
    }
}

// Warp-level sum reduction for float or double
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* smem_warp) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        smem_warp[wid] = val;
    }
    __syncthreads();
    scalar_t out = scalar_t(0);
    if (wid == 0) {
        out = (lane < nwarps) ? smem_warp[lane] : scalar_t(0);
        out = warp_reduce_sum(out);
    }
    __syncthreads();
    return out;
}

// ============================================================================
// KERNEL 1: Fused Binary AR(1) Laplace Evaluation (1 block per active trial)
// ============================================================================
template <typename scalar_t, int T>
__global__ void laplace_ar1_eval_kernel(
    const scalar_t* __restrict__ params,   // [M, D], D = p + 3
    const int64_t* __restrict__ idx,       // [M]
    const scalar_t* __restrict__ Y_bit,    // [B, I, T]
    const scalar_t* __restrict__ X_itp,    // [I, T, p]
    const scalar_t* __restrict__ K_it,     // [I, T]
    const scalar_t* __restrict__ u_in,     // [M, I, T]
    scalar_t* __restrict__ f_out,          // [M]
    scalar_t* __restrict__ grad_out,       // [M, D]
    scalar_t* __restrict__ u_out,          // [M, I, T]
    int M,
    int I_c,
    int p_dim,
    int n_inner,
    double inner_tol
) {
    constexpr int SYM_T = T * (T + 1) / 2;
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D = p_dim + 3;

    __shared__ scalar_t beta_s[MAX_P];
    __shared__ scalar_t Om_s[T][T];
    __shared__ scalar_t Om_inv_s[T][T];
    __shared__ scalar_t R_s[T][T];
    __shared__ scalar_t dR_s[T][T];
    __shared__ scalar_t vc_s, vcp_s, inv_s_s, logdet_Om_s;
    __shared__ int bad_block;
    __shared__ scalar_t smem_warp[8];
    __shared__ scalar_t sym_sum_s[SYM_T];

    if (tid == 0) {
        bad_block = 0;
        const scalar_t* pm = params + (int64_t)m * D;
        for (int q = 0; q < p_dim; ++q) {
            beta_s[q] = pm[q];
        }
        scalar_t p1 = clamp_val(pm[p_dim], scalar_t(-16.0), scalar_t(16.0));
        scalar_t p2 = clamp_val(pm[p_dim + 1], scalar_t(-16.0), scalar_t(16.0));
        scalar_t p3 = pm[p_dim + 2];

        scalar_t vc = exp(scalar_t(2.0) * p1);
        scalar_t vcp = exp(scalar_t(2.0) * p2);
        scalar_t inv_s = scalar_t(1.0) / sqrt(scalar_t(1.0) + p3 * p3);
        scalar_t rho = p3 * inv_s;

        vc_s = vc;
        vcp_s = vcp;
        inv_s_s = inv_s;

        scalar_t pow_rho[T];
        pow_rho[0] = scalar_t(1.0);
        for (int d = 1; d < T; ++d) {
            pow_rho[d] = pow_rho[d - 1] * rho;
        }

        scalar_t L_Om[SYM_T];
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < T; ++s) {
                int d = t >= s ? (t - s) : (s - t);
                scalar_t r_ts = pow_rho[d];
                R_s[t][s] = r_ts;
                dR_s[t][s] = (d > 0) ? (scalar_t(d) * pow_rho[d - 1]) : scalar_t(0.0);
                scalar_t om_ts = vc + vcp * r_ts + (t == s ? scalar_t(1e-13) : scalar_t(0.0));
                Om_s[t][s] = om_ts;
                if (t >= s) {
                    L_Om[sym_idx(t, s)] = om_ts;
                }
            }
        }

        bool ok_om = cholesky_lower_inplace<scalar_t, T>(L_Om);
        if (!ok_om) {
            bad_block = 1;
        } else {
            scalar_t ld = scalar_t(0.0);
            scalar_t tny = tiny_val<scalar_t>();
            for (int t = 0; t < T; ++t) {
                scalar_t d_val = L_Om[sym_idx(t, t)];
                ld += log(d_val > tny ? d_val : tny);
            }
            logdet_Om_s = scalar_t(2.0) * ld;

            invert_lower_inplace<scalar_t, T>(L_Om);
            // Om_inv = L_Om^{-T} * L_Om^{-1}
            for (int t = 0; t < T; ++t) {
                for (int s = 0; s <= t; ++s) {
                    scalar_t acc = scalar_t(0.0);
                    for (int r = t; r < T; ++r) {
                        acc += L_Om[sym_idx(r, t)] * L_Om[sym_idx(r, s)];
                    }
                    Om_inv_s[t][s] = acc;
                    Om_inv_s[s][t] = acc;
                }
            }
        }
    }
    __syncthreads();

    if (bad_block != 0) {
        if (tid == 0) {
            f_out[m] = inf_val<scalar_t>();
            scalar_t* gm = grad_out + (int64_t)m * D;
            for (int q = 0; q < D; ++q) gm[q] = scalar_t(0.0);
        }
        return;
    }

    const int64_t trial_b = idx[m];
    const scalar_t tny = tiny_val<scalar_t>();
    const scalar_t tol_s = static_cast<scalar_t>(inner_tol);

    scalar_t f_loc = scalar_t(0.0);
    scalar_t gbeta_loc[MAX_P];
    for (int q = 0; q < p_dim; ++q) gbeta_loc[q] = scalar_t(0.0);
    scalar_t sym_loc[SYM_T];
    for (int k = 0; k < SYM_T; ++k) sym_loc[k] = scalar_t(0.0);
    int bad_loc = 0;

    for (int i = tid; i < I_c; i += blockDim.x) {
        scalar_t y[T], k_v[T], ef[T], u[T];
        const scalar_t* y_ptr = Y_bit + (trial_b * I_c + i) * T;
        const scalar_t* k_ptr = K_it + (int64_t)i * T;
        const scalar_t* x_ptr = X_itp + (int64_t)i * T * p_dim;
        const scalar_t* u_in_ptr = u_in + ((int64_t)m * I_c + i) * T;

        for (int t = 0; t < T; ++t) {
            y[t] = y_ptr[t];
            k_v[t] = k_ptr[t];
            u[t] = u_in_ptr[t];
            scalar_t acc = scalar_t(0.0);
            const scalar_t* x_it = x_ptr + t * p_dim;
            for (int q = 0; q < p_dim; ++q) {
                acc += x_it[q] * beta_s[q];
            }
            ef[t] = acc;
        }

        // Inner Newton loop in registers
        scalar_t L[SYM_T];
        scalar_t g[T], du[T];
        for (int it = 0; it < n_inner; ++it) {
            scalar_t W[T];
            for (int t = 0; t < T; ++t) {
                scalar_t eta = clamp_val(ef[t] + u[t], scalar_t(-28.0), scalar_t(28.0));
                scalar_t mu = sigmoid_val(eta);
                scalar_t w = k_v[t] * mu * (scalar_t(1.0) - mu);
                W[t] = w > scalar_t(1e-15) ? w : scalar_t(1e-15);

                scalar_t u_om = scalar_t(0.0);
                for (int s = 0; s < T; ++s) {
                    u_om += Om_inv_s[t][s] * u[s];
                }
                g[t] = (y[t] - k_v[t] * mu) - u_om;
            }

            for (int r = 0; r < T; ++r) {
                for (int c = 0; c < r; ++c) {
                    L[sym_idx(r, c)] = Om_inv_s[r][c];
                }
                L[sym_idx(r, r)] = Om_inv_s[r][r] + W[r];
            }

            bool ok_hu = cholesky_lower_inplace<scalar_t, T>(L);
            if (ok_hu) {
                cholesky_solve_vec<scalar_t, T>(L, g, du);
            } else {
                for (int t = 0; t < T; ++t) du[t] = g[t];
            }

            scalar_t max_du = scalar_t(0.0);
            for (int t = 0; t < T; ++t) {
                u[t] += du[t];
                scalar_t ad = fabs(du[t]);
                if (ad > max_du) max_du = ad;
            }
            if (max_du < tol_s) {
                break;
            }
        }

        // Write converged u back
        scalar_t* u_out_ptr = u_out + ((int64_t)m * I_c + i) * T;
        for (int t = 0; t < T; ++t) {
            u_out_ptr[t] = u[t];
        }

        // Final evaluation & exact adjoint at converged mode u
        scalar_t eta[T], mu[T], W[T];
        scalar_t ll_y = scalar_t(0.0);
        scalar_t quad_u = scalar_t(0.0);
        for (int t = 0; t < T; ++t) {
            eta[t] = clamp_val(ef[t] + u[t], scalar_t(-28.0), scalar_t(28.0));
            mu[t] = sigmoid_val(eta[t]);
            scalar_t w = k_v[t] * mu[t] * (scalar_t(1.0) - mu[t]);
            W[t] = w > scalar_t(1e-15) ? w : scalar_t(1e-15);
            ll_y += y[t] * eta[t] - k_v[t] * log1p_exp_val(eta[t]);

            scalar_t u_om = scalar_t(0.0);
            for (int s = 0; s < T; ++s) {
                u_om += Om_inv_s[t][s] * u[s];
            }
            quad_u += scalar_t(0.5) * u[t] * u_om;
        }

        for (int r = 0; r < T; ++r) {
            for (int c = 0; c < r; ++c) {
                L[sym_idx(r, c)] = Om_inv_s[r][c];
            }
            L[sym_idx(r, r)] = Om_inv_s[r][r] + W[r];
        }

        bool ok_hu = cholesky_lower_inplace<scalar_t, T>(L);
        if (!ok_hu) {
            bad_loc = 1;
            continue;
        }

        scalar_t ld_hu = scalar_t(0.0);
        for (int t = 0; t < T; ++t) {
            scalar_t d_val = L[sym_idx(t, t)];
            ld_hu += log(d_val > tny ? d_val : tny);
        }
        f_loc += -ll_y + quad_u + ld_hu;  // 0.5 * (2 * ld_hu) = ld_hu

        // Invert L in-place -> L^{-1}
        invert_lower_inplace<scalar_t, T>(L);

        // Compute v[t] = 0.5 * Hu_inv[t,t] * W[t] * (1 - 2*mu[t])
        scalar_t v[T], tmp[T], lam[T];
        for (int t = 0; t < T; ++t) {
            scalar_t h_tt = scalar_t(0.0);
            for (int r = t; r < T; ++r) {
                scalar_t l_rt = L[sym_idx(r, t)];
                h_tt += l_rt * l_rt;
            }
            v[t] = scalar_t(0.5) * h_tt * W[t] * (scalar_t(1.0) - scalar_t(2.0) * mu[t]);
        }

        // lam = L^{-T} * (L^{-1} * v)
        for (int r = 0; r < T; ++r) {
            scalar_t acc = scalar_t(0.0);
            for (int c = 0; c <= r; ++c) {
                acc += L[sym_idx(r, c)] * v[c];
            }
            tmp[r] = acc;
        }
        for (int t = 0; t < T; ++t) {
            scalar_t acc = scalar_t(0.0);
            for (int r = t; r < T; ++r) {
                acc += L[sym_idx(r, t)] * tmp[r];
            }
            lam[t] = acc;
        }

        // Accumulate grad_beta
        for (int t = 0; t < T; ++t) {
            scalar_t gef = -(y[t] - k_v[t] * mu[t]) + v[t] - W[t] * lam[t];
            const scalar_t* x_it = x_ptr + t * p_dim;
            for (int q = 0; q < p_dim; ++q) {
                gbeta_loc[q] += x_it[q] * gef;
            }
        }

        // Accumulate symmetric G_Ominv components: 0.5*(u*u^T + Hu_inv - lam*u^T - u*lam^T)
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s <= t; ++s) {
                scalar_t hu_ts = scalar_t(0.0);
                for (int r = t; r < T; ++r) {
                    hu_ts += L[sym_idx(r, t)] * L[sym_idx(r, s)];
                }
                sym_loc[sym_idx(t, s)] += scalar_t(0.5) * (
                    u[t] * u[s] + hu_ts - lam[t] * u[s] - u[t] * lam[s]
                );
            }
        }
    }

    // Reduce bad flag
    if (bad_loc != 0) {
        atomicOr(&bad_block, 1);
    }
    __syncthreads();
    if (bad_block != 0) {
        if (tid == 0) {
            f_out[m] = inf_val<scalar_t>();
            scalar_t* gm = grad_out + (int64_t)m * D;
            for (int q = 0; q < D; ++q) gm[q] = scalar_t(0.0);
        }
        return;
    }

    // Block reduce f_loc
    scalar_t f_sum = block_reduce_sum(f_loc, smem_warp);
    if (tid == 0) {
        f_out[m] = f_sum + scalar_t(0.5) * scalar_t(I_c) * logdet_Om_s;
    }

    // Block reduce grad_beta[0..p_dim-1]
    scalar_t* gm = grad_out + (int64_t)m * D;
    for (int q = 0; q < p_dim; ++q) {
        scalar_t gq = block_reduce_sum(gbeta_loc[q], smem_warp);
        if (tid == 0) gm[q] = gq;
    }

    // Block reduce symmetric matrix sum
    for (int k = 0; k < SYM_T; ++k) {
        scalar_t sk = block_reduce_sum(sym_loc[k], smem_warp);
        if (tid == 0) sym_sum_s[k] = sk;
    }

    if (tid == 0) {
        scalar_t G_Ominv[T][T];
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s <= t; ++s) {
                scalar_t val = sym_sum_s[sym_idx(t, s)] - scalar_t(0.5) * scalar_t(I_c) * Om_s[t][s];
                G_Ominv[t][s] = val;
                G_Ominv[s][t] = val;
            }
        }

        // G_Om = -(Om_inv * G_Ominv * Om_inv)
        scalar_t tmpM[T][T];
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < T; ++s) {
                scalar_t acc = scalar_t(0.0);
                for (int r = 0; r < T; ++r) {
                    acc += Om_inv_s[t][r] * G_Ominv[r][s];
                }
                tmpM[t][s] = acc;
            }
        }

        scalar_t sum_gom = scalar_t(0.0);
        scalar_t sum_gom_R = scalar_t(0.0);
        scalar_t sum_gom_dR = scalar_t(0.0);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < T; ++s) {
                scalar_t acc = scalar_t(0.0);
                for (int r = 0; r < T; ++r) {
                    acc += tmpM[t][r] * Om_inv_s[r][s];
                }
                scalar_t gom = -acc;
                sum_gom += gom;
                sum_gom_R += gom * R_s[t][s];
                sum_gom_dR += gom * dR_s[t][s];
            }
        }

        gm[p_dim] = scalar_t(2.0) * vc_s * sum_gom;
        gm[p_dim + 1] = scalar_t(2.0) * vcp_s * sum_gom_R;
        gm[p_dim + 2] = vcp_s * inv_s_s * inv_s_s * inv_s_s * sum_gom_dR;
    }
}

// ============================================================================
// KERNEL 2: Fused Continuous AR(1) REML/ML Evaluation (1 thread per trial)
// ============================================================================
template <typename scalar_t, int T, int P>
__global__ void cont_ar1_eval_kernel(
    const scalar_t* __restrict__ psi_sub,  // [M, 4]
    const int64_t* __restrict__ idx,       // [M]
    const scalar_t* __restrict__ S_YY,     // [B, T, T]
    const scalar_t* __restrict__ G_YX,     // [B, T, T, P]
    const scalar_t* __restrict__ M_XX,     // [T, T, P, P]
    const scalar_t* __restrict__ K_t,      // [T]
    scalar_t* __restrict__ f_out,          // [M]
    scalar_t* __restrict__ grad_out,       // [M, 4]
    scalar_t* __restrict__ beta_out,       // [M, P] (optional, non-null if need_aux)
    scalar_t* __restrict__ ainv_last_out,  // [M]    (optional, non-null if need_aux)
    int M,
    int I_c,
    bool reml,
    bool need_aux
) {
    constexpr int SYM_T = T * (T + 1) / 2;
    constexpr int SYM_P = P * (P + 1) / 2;

    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const scalar_t tny = tiny_val<scalar_t>();
    const scalar_t* pm = psi_sub + (int64_t)m * 4;
    scalar_t p0 = clamp_val(pm[0], scalar_t(-35.0), scalar_t(35.0));
    scalar_t p1 = clamp_val(pm[1], scalar_t(-18.0), scalar_t(18.0));
    scalar_t p2 = clamp_val(pm[2], scalar_t(-18.0), scalar_t(18.0));
    scalar_t p3 = pm[3];

    scalar_t exp_p0 = exp(p0);
    scalar_t vc = exp(scalar_t(2.0) * p1);
    scalar_t vcp = exp(scalar_t(2.0) * p2);
    scalar_t inv_s = scalar_t(1.0) / sqrt(scalar_t(1.0) + p3 * p3);
    scalar_t rho = p3 * inv_s;

    scalar_t pow_rho[T];
    pow_rho[0] = scalar_t(1.0);
    for (int d = 1; d < T; ++d) pow_rho[d] = pow_rho[d - 1] * rho;

    scalar_t ve[T];
    for (int t = 0; t < T; ++t) ve[t] = exp_p0 / K_t[t];

    scalar_t L_Sig[SYM_T];
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s <= t; ++s) {
            int d = t - s;
            scalar_t v = vc + vcp * pow_rho[d] + (t == s ? ve[t] : scalar_t(0.0));
            L_Sig[sym_idx(t, s)] = v;
        }
    }

    bool ok_s = cholesky_lower_inplace<scalar_t, T>(L_Sig);
    if (!ok_s) {
        f_out[m] = inf_val<scalar_t>();
        for (int j = 0; j < 4; ++j) grad_out[(int64_t)m * 4 + j] = scalar_t(0.0);
        if (need_aux) {
            for (int q = 0; q < P; ++q) beta_out[(int64_t)m * P + q] = scalar_t(0.0);
            ainv_last_out[m] = scalar_t(0.0);
        }
        return;
    }

    scalar_t logdet_Sig = scalar_t(0.0);
    for (int t = 0; t < T; ++t) {
        scalar_t d_val = L_Sig[sym_idx(t, t)];
        logdet_Sig += log(d_val > tny ? d_val : tny);
    }
    logdet_Sig *= scalar_t(2.0);

    invert_lower_inplace<scalar_t, T>(L_Sig);
    scalar_t Sinv[T][T];
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s <= t; ++s) {
            scalar_t acc = scalar_t(0.0);
            for (int r = t; r < T; ++r) acc += L_Sig[sym_idx(r, t)] * L_Sig[sym_idx(r, s)];
            Sinv[t][s] = acc;
            Sinv[s][t] = acc;
        }
    }

    // A = sum_{t,s} Sinv[t,s] * M_XX[t,s,q1,q2]
    scalar_t L_A[SYM_P];
    for (int k = 0; k < SYM_P; ++k) L_A[k] = scalar_t(0.0);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            scalar_t sv = Sinv[t][s];
            const scalar_t* mxx_ts = M_XX + ((t * T + s) * P) * P;
            for (int q1 = 0; q1 < P; ++q1) {
                const scalar_t* row_q1 = mxx_ts + q1 * P;
                for (int q2 = 0; q2 <= q1; ++q2) {
                    L_A[sym_idx(q1, q2)] += sv * row_q1[q2];
                }
            }
        }
    }

    bool ok_a = cholesky_lower_inplace<scalar_t, P>(L_A);
    if (!ok_a) {
        f_out[m] = inf_val<scalar_t>();
        for (int j = 0; j < 4; ++j) grad_out[(int64_t)m * 4 + j] = scalar_t(0.0);
        if (need_aux) {
            for (int q = 0; q < P; ++q) beta_out[(int64_t)m * P + q] = scalar_t(0.0);
            ainv_last_out[m] = scalar_t(0.0);
        }
        return;
    }

    scalar_t logdet_A = scalar_t(0.0);
    for (int q = 0; q < P; ++q) {
        scalar_t d_val = L_A[sym_idx(q, q)];
        logdet_A += log(d_val > tny ? d_val : tny);
    }
    logdet_A *= scalar_t(2.0);

    const int64_t b_idx = idx[m];
    const scalar_t* syy_m = S_YY + b_idx * T * T;
    const scalar_t* gyx_m = G_YX + b_idx * T * T * P;

    scalar_t rhs[P];
    for (int q = 0; q < P; ++q) rhs[q] = scalar_t(0.0);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            scalar_t sv = Sinv[t][s];
            const scalar_t* g_ts = gyx_m + (t * T + s) * P;
            for (int q = 0; q < P; ++q) {
                rhs[q] += sv * g_ts[q];
            }
        }
    }

    scalar_t beta[P];
    cholesky_solve_vec<scalar_t, P>(L_A, rhs, beta);

    // quad = tr(Sinv * S_YY) - beta^T rhs
    scalar_t tr_syy = scalar_t(0.0);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            tr_syy += Sinv[t][s] * syy_m[t * T + s];
        }
    }
    scalar_t br = scalar_t(0.0);
    for (int q = 0; q < P; ++q) br += beta[q] * rhs[q];
    scalar_t quad = tr_syy - br;

    if (reml) {
        f_out[m] = scalar_t(0.5) * (scalar_t(I_c) * logdet_Sig + logdet_A + quad);
    } else {
        f_out[m] = scalar_t(0.5) * (scalar_t(I_c) * logdet_Sig + quad);
    }

    // Invert L_A -> Ainv if REML or need_aux
    scalar_t C_mat[P][P];
    for (int q1 = 0; q1 < P; ++q1) {
        for (int q2 = 0; q2 < P; ++q2) {
            C_mat[q1][q2] = beta[q1] * beta[q2];
        }
    }

    scalar_t ainv_last = scalar_t(0.0);
    if (reml || need_aux) {
        invert_lower_inplace<scalar_t, P>(L_A);
        scalar_t l_last = L_A[sym_idx(P - 1, P - 1)];
        ainv_last = l_last * l_last;
        if (reml) {
            for (int q1 = 0; q1 < P; ++q1) {
                for (int q2 = 0; q2 <= q1; ++q2) {
                    scalar_t acc = scalar_t(0.0);
                    for (int r = q1; r < P; ++r) {
                        acc += L_A[sym_idx(r, q1)] * L_A[sym_idx(r, q2)];
                    }
                    C_mat[q1][q2] += acc;
                    if (q1 != q2) C_mat[q2][q1] += acc;
                }
            }
        }
    }

    if (need_aux) {
        for (int q = 0; q < P; ++q) beta_out[(int64_t)m * P + q] = beta[q];
        ainv_last_out[m] = ainv_last;
    }

    // Build mid[t, s] = S_YY[t,s] - XY_b[t,s] - XY_b[s,t] + sum_{q1,q2} M_XX[t,s,q1,q2] * C_mat[q1,q2]
    scalar_t xy_b[T][T];
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            const scalar_t* g_ts = gyx_m + (t * T + s) * P;
            scalar_t acc = scalar_t(0.0);
            for (int q = 0; q < P; ++q) acc += g_ts[q] * beta[q];
            xy_b[t][s] = acc;
        }
    }

    scalar_t mid[T][T];
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s <= t; ++s) {
            const scalar_t* mxx_ts = M_XX + ((t * T + s) * P) * P;
            scalar_t xcx = scalar_t(0.0);
            for (int q1 = 0; q1 < P; ++q1) {
                const scalar_t* row_q1 = mxx_ts + q1 * P;
                for (int q2 = 0; q2 < P; ++q2) {
                    xcx += row_q1[q2] * C_mat[q1][q2];
                }
            }
            scalar_t mv = syy_m[t * T + s] - xy_b[t][s] - xy_b[s][t] + xcx;
            mid[t][s] = mv;
            mid[s][t] = mv;
        }
    }

    // G_Sig = 0.5 * (I_c * Sinv - Sinv * mid * Sinv)
    scalar_t tmpM[T][T];
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            scalar_t acc = scalar_t(0.0);
            for (int r = 0; r < T; ++r) acc += Sinv[t][r] * mid[r][s];
            tmpM[t][s] = acc;
        }
    }

    scalar_t g0 = scalar_t(0.0);
    scalar_t sum_gs = scalar_t(0.0);
    scalar_t sum_gs_R = scalar_t(0.0);
    scalar_t sum_gs_dR = scalar_t(0.0);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < T; ++s) {
            scalar_t acc = scalar_t(0.0);
            for (int r = 0; r < T; ++r) acc += tmpM[t][r] * Sinv[r][s];
            scalar_t gsig = scalar_t(0.5) * (scalar_t(I_c) * Sinv[t][s] - acc);
            if (t == s) g0 += gsig * ve[t];
            sum_gs += gsig;
            int d = t >= s ? (t - s) : (s - t);
            sum_gs_R += gsig * pow_rho[d];
            if (d > 0) sum_gs_dR += gsig * (scalar_t(d) * pow_rho[d - 1]);
        }
    }

    scalar_t* gm = grad_out + (int64_t)m * 4;
    gm[0] = g0;
    gm[1] = scalar_t(2.0) * vc * sum_gs;
    gm[2] = scalar_t(2.0) * vcp * sum_gs_R;
    gm[3] = vcp * inv_s * inv_s * inv_s * sum_gs_dR;
}

// ============================================================================
// KERNEL 3: Fused Binary RI PIRLS Solver (1 block per active trial)
// Matches lme4::glmer(..., family = binomial(), nAGQ = 0)
// ============================================================================
template <typename scalar_t, int T, int P>
__global__ void pirls_binary_kernel(
    const scalar_t* __restrict__ theta_sub,  // [M]
    const int64_t* __restrict__ idx,         // [M]
    const scalar_t* __restrict__ Y_bit,      // [B, I, T]
    const scalar_t* __restrict__ K_it,       // [I, T]
    const scalar_t* __restrict__ tx_it,      // [I, T]
    const scalar_t* __restrict__ beta_in,    // [M, P]
    const scalar_t* __restrict__ z_in,       // [M, I]
    scalar_t* __restrict__ dev_out,          // [M]
    scalar_t* __restrict__ beta_tx_out,      // [M]
    scalar_t* __restrict__ se_tx_out,        // [M]
    bool* __restrict__ valid_out,            // [M]
    scalar_t* __restrict__ beta_out,         // [M, P]
    scalar_t* __restrict__ z_out,            // [M, I]
    int M,
    int I_c,
    bool adjust_period,
    int max_iter,
    double tol
) {
    constexpr int SYM_P = P * (P + 1) / 2;
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int64_t b_idx = idx[m];
    const scalar_t th = theta_sub[m];
    const scalar_t th2 = th * th;
    const scalar_t tny = tiny_val<scalar_t>();
    const scalar_t tol_s = static_cast<scalar_t>(tol);

    __shared__ scalar_t beta_s[P];
    __shared__ scalar_t dbeta_s[P];
    __shared__ scalar_t A_s[SYM_P];
    __shared__ scalar_t bb_s[P];
    __shared__ scalar_t step_s;
    __shared__ scalar_t pdev_s;
    __shared__ scalar_t ndec_s;
    __shared__ int status_s;  // 0 = active, 1 = converged, 2 = failed
    __shared__ scalar_t smem_warp[8];

    if (tid == 0) {
        const scalar_t* bin_m = beta_in + (int64_t)m * P;
        for (int q = 0; q < P; ++q) beta_s[q] = bin_m[q];
        status_s = 0;
    }
    __syncthreads();

    // Support arbitrary I_c via up to 4 clusters per thread (covers up to I_c = 512 with block=128)
    constexpr int MAX_LOCAL_I = 8;
    int n_loc = 0;
    int i_list[MAX_LOCAL_I];
    scalar_t z_loc[MAX_LOCAL_I];
    scalar_t dz_loc[MAX_LOCAL_I];
    for (int i = tid; i < I_c; i += blockDim.x) {
        if (n_loc < MAX_LOCAL_I) {
            i_list[n_loc] = i;
            z_loc[n_loc] = z_in[(int64_t)m * I_c + i];
            n_loc++;
        }
    }

    // Initial penalized deviance
    scalar_t pdev_loc = scalar_t(0.0);
    for (int li = 0; li < n_loc; ++li) {
        int i = i_list[li];
        const scalar_t* y_ptr = Y_bit + (b_idx * I_c + i) * T;
        const scalar_t* k_ptr = K_it + (int64_t)i * T;
        const scalar_t* tx_ptr = tx_it + (int64_t)i * T;
        scalar_t zv = z_loc[li];
        scalar_t ll = scalar_t(0.0);
        for (int t = 0; t < T; ++t) {
            scalar_t ef = beta_s[0] + beta_s[P - 1] * tx_ptr[t];
            if (adjust_period && t > 0) ef += beta_s[t];
            scalar_t eta = clamp_val(ef + th * zv, scalar_t(-30.0), scalar_t(30.0));
            ll += y_ptr[t] * eta - k_ptr[t] * log1p_exp_val(eta);
        }
        pdev_loc += scalar_t(-2.0) * ll + zv * zv;
    }
    scalar_t pdev_tot = block_reduce_sum(pdev_loc, smem_warp);
    if (tid == 0) pdev_s = pdev_tot;
    __syncthreads();

    for (int it = 0; it < max_iter; ++it) {
        if (status_s != 0) break;

        scalar_t A_loc[SYM_P];
        scalar_t bb_loc[P];
        for (int k = 0; k < SYM_P; ++k) A_loc[k] = scalar_t(0.0);
        for (int q = 0; q < P; ++q) bb_loc[q] = scalar_t(0.0);
        scalar_t rz2_over_dz_loc = scalar_t(0.0);

        // Save per-cluster intermediate values for dz calculation
        scalar_t rz_cache[MAX_LOCAL_I];
        scalar_t dz_denom_cache[MAX_LOCAL_I];
        scalar_t swx_cache[MAX_LOCAL_I][P];

        for (int li = 0; li < n_loc; ++li) {
            int i = i_list[li];
            const scalar_t* y_ptr = Y_bit + (b_idx * I_c + i) * T;
            const scalar_t* k_ptr = K_it + (int64_t)i * T;
            const scalar_t* tx_ptr = tx_it + (int64_t)i * T;
            scalar_t zv = z_loc[li];

            scalar_t w_clus = scalar_t(0.0);
            scalar_t w_tx = scalar_t(0.0);
            scalar_t w_tx2 = scalar_t(0.0);
            scalar_t s_g = scalar_t(0.0);
            scalar_t g_tx = scalar_t(0.0);
            scalar_t swx[P];
            scalar_t rhb[P];
            for (int q = 0; q < P; ++q) {
                swx[q] = scalar_t(0.0);
                rhb[q] = scalar_t(0.0);
            }

            for (int t = 0; t < T; ++t) {
                scalar_t txv = tx_ptr[t];
                scalar_t ef = beta_s[0] + beta_s[P - 1] * txv;
                if (adjust_period && t > 0) ef += beta_s[t];
                scalar_t eta = clamp_val(ef + th * zv, scalar_t(-30.0), scalar_t(30.0));
                scalar_t mu = sigmoid_val(eta);
                scalar_t w = k_ptr[t] * mu * (scalar_t(1.0) - mu);
                if (w < scalar_t(1e-15)) w = scalar_t(1e-15);
                scalar_t g = y_ptr[t] - k_ptr[t] * mu;

                w_clus += w;
                w_tx += w * txv;
                w_tx2 += w * txv * txv;
                s_g += g;
                g_tx += g * txv;
                if (adjust_period && t > 0) {
                    swx[t] = w;
                    rhb[t] = g;
                    A_loc[sym_idx(t, 0)] += w;
                    A_loc[sym_idx(t, t)] += w;
                    A_loc[sym_idx(P - 1, t)] += w * txv;
                }
            }
            swx[0] = w_clus;
            swx[P - 1] = w_tx;
            rhb[0] = s_g;
            rhb[P - 1] = g_tx;

            A_loc[sym_idx(0, 0)] += w_clus;
            A_loc[sym_idx(P - 1, 0)] += w_tx;
            A_loc[sym_idx(P - 1, P - 1)] += w_tx2;

            scalar_t D_z = scalar_t(1.0) + th2 * w_clus;
            scalar_t rhs_z = th * s_g - zv;
            scalar_t sc_z = th2 / D_z;
            scalar_t sc_b = th * rhs_z / D_z;

            rz_cache[li] = rhs_z;
            dz_denom_cache[li] = D_z;
            for (int q = 0; q < P; ++q) swx_cache[li][q] = swx[q];
            rz2_over_dz_loc += (rhs_z * rhs_z) / D_z;

            for (int r = 0; r < P; ++r) {
                bb_loc[r] += rhb[r] - sc_b * swx[r];
                for (int c = 0; c <= r; ++c) {
                    A_loc[sym_idx(r, c)] -= sc_z * swx[r] * swx[c];
                }
            }
        }

        for (int k = 0; k < SYM_P; ++k) {
            scalar_t v = block_reduce_sum(A_loc[k], smem_warp);
            if (tid == 0) A_s[k] = v;
        }
        for (int q = 0; q < P; ++q) {
            scalar_t v = block_reduce_sum(bb_loc[q], smem_warp);
            if (tid == 0) bb_s[q] = v;
        }
        scalar_t rz2_sum = block_reduce_sum(rz2_over_dz_loc, smem_warp);

        if (tid == 0) {
            bool ok = cholesky_lower_inplace<scalar_t, P>(A_s);
            if (!ok) {
                status_s = 2;
            } else {
                cholesky_solve_vec<scalar_t, P>(A_s, bb_s, dbeta_s);
                scalar_t dot_b = scalar_t(0.0);
                for (int q = 0; q < P; ++q) dot_b += bb_s[q] * dbeta_s[q];
                scalar_t ndec2 = rz2_sum + dot_b;
                if (ndec2 < scalar_t(0.0)) ndec2 = scalar_t(0.0);
                step_s = scalar_t(1.0);
                ndec_s = sqrt(ndec2);
            }
        }
        __syncthreads();
        if (status_s != 0) break;

        for (int li = 0; li < n_loc; ++li) {
            scalar_t swx_db = scalar_t(0.0);
            for (int q = 0; q < P; ++q) swx_db += swx_cache[li][q] * dbeta_s[q];
            dz_loc[li] = (rz_cache[li] - th * swx_db) / dz_denom_cache[li];
        }

        // Up to 5 step-halving evaluations (step = 1, 0.5, 0.25, 0.125, 0.0625)
        scalar_t ndec_val = ndec_s;
        scalar_t cur_step = scalar_t(1.0);
        for (int h = 0; h < 5; ++h) {
            scalar_t try_loc = scalar_t(0.0);
            for (int li = 0; li < n_loc; ++li) {
                int i = i_list[li];
                const scalar_t* y_ptr = Y_bit + (b_idx * I_c + i) * T;
                const scalar_t* k_ptr = K_it + (int64_t)i * T;
                const scalar_t* tx_ptr = tx_it + (int64_t)i * T;
                scalar_t zv = z_loc[li] + cur_step * dz_loc[li];
                scalar_t ll = scalar_t(0.0);
                for (int t = 0; t < T; ++t) {
                    scalar_t txv = tx_ptr[t];
                    scalar_t ef = (beta_s[0] + cur_step * dbeta_s[0])
                                + (beta_s[P - 1] + cur_step * dbeta_s[P - 1]) * txv;
                    if (adjust_period && t > 0) ef += (beta_s[t] + cur_step * dbeta_s[t]);
                    scalar_t eta = clamp_val(ef + th * zv, scalar_t(-30.0), scalar_t(30.0));
                    ll += y_ptr[t] * eta - k_ptr[t] * log1p_exp_val(eta);
                }
                try_loc += scalar_t(-2.0) * ll + zv * zv;
            }
            scalar_t try_tot = block_reduce_sum(try_loc, smem_warp);
            if (tid == 0) {
                if (try_tot <= pdev_s + scalar_t(1e-6) * (scalar_t(1.0) + fabs(pdev_s)) || h == 4) {
                    pdev_s = try_tot;
                    step_s = cur_step;
                    status_s = (ndec_val < tol_s) ? 1 : 0;
                } else {
                    step_s = scalar_t(-1.0);
                }
            }
            __syncthreads();
            if (step_s > scalar_t(0.0)) break;
            cur_step *= scalar_t(0.5);
        }

        // Apply accepted step
        scalar_t acc_step = step_s;
        if (tid == 0) {
            for (int q = 0; q < P; ++q) beta_s[q] += acc_step * dbeta_s[q];
        }
        for (int li = 0; li < n_loc; ++li) {
            z_loc[li] += acc_step * dz_loc[li];
        }
        __syncthreads();
    }

    // Write final z_out
    for (int li = 0; li < n_loc; ++li) {
        z_out[(int64_t)m * I_c + i_list[li]] = z_loc[li];
    }

    if (status_s == 2) {
        if (tid == 0) {
            dev_out[m] = inf_val<scalar_t>();
            beta_tx_out[m] = scalar_t(0.0);
            se_tx_out[m] = scalar_t(0.0);
            valid_out[m] = false;
            for (int q = 0; q < P; ++q) beta_out[(int64_t)m * P + q] = beta_s[q];
        }
        return;
    }

    // Final Laplace deviance & SE at converged mode
    scalar_t A_loc[SYM_P];
    for (int k = 0; k < SYM_P; ++k) A_loc[k] = scalar_t(0.0);
    scalar_t log_dz_loc = scalar_t(0.0);

    for (int li = 0; li < n_loc; ++li) {
        int i = i_list[li];
        const scalar_t* k_ptr = K_it + (int64_t)i * T;
        const scalar_t* tx_ptr = tx_it + (int64_t)i * T;
        scalar_t zv = z_loc[li];

        scalar_t w_clus = scalar_t(0.0);
        scalar_t w_tx = scalar_t(0.0);
        scalar_t w_tx2 = scalar_t(0.0);
        scalar_t swx[P];
        for (int q = 0; q < P; ++q) swx[q] = scalar_t(0.0);

        for (int t = 0; t < T; ++t) {
            scalar_t txv = tx_ptr[t];
            scalar_t ef = beta_s[0] + beta_s[P - 1] * txv;
            if (adjust_period && t > 0) ef += beta_s[t];
            scalar_t eta = clamp_val(ef + th * zv, scalar_t(-30.0), scalar_t(30.0));
            scalar_t mu = sigmoid_val(eta);
            scalar_t w = k_ptr[t] * mu * (scalar_t(1.0) - mu);
            if (w < scalar_t(1e-15)) w = scalar_t(1e-15);

            w_clus += w;
            w_tx += w * txv;
            w_tx2 += w * txv * txv;
            if (adjust_period && t > 0) {
                swx[t] = w;
                A_loc[sym_idx(t, 0)] += w;
                A_loc[sym_idx(t, t)] += w;
                A_loc[sym_idx(P - 1, t)] += w * txv;
            }
        }
        swx[0] = w_clus;
        swx[P - 1] = w_tx;
        A_loc[sym_idx(0, 0)] += w_clus;
        A_loc[sym_idx(P - 1, 0)] += w_tx;
        A_loc[sym_idx(P - 1, P - 1)] += w_tx2;

        scalar_t D_z = scalar_t(1.0) + th2 * w_clus;
        log_dz_loc += log(D_z > tny ? D_z : tny);
        scalar_t sc_z = th2 / D_z;
        for (int r = 0; r < P; ++r) {
            for (int c = 0; c <= r; ++c) {
                A_loc[sym_idx(r, c)] -= sc_z * swx[r] * swx[c];
            }
        }
    }

    for (int k = 0; k < SYM_P; ++k) {
        scalar_t v = block_reduce_sum(A_loc[k], smem_warp);
        if (tid == 0) A_s[k] = v;
    }
    scalar_t log_dz_tot = block_reduce_sum(log_dz_loc, smem_warp);

    if (tid == 0) {
        for (int q = 0; q < P; ++q) beta_out[(int64_t)m * P + q] = beta_s[q];
        bool ok = cholesky_lower_inplace<scalar_t, P>(A_s);
        scalar_t lap_dev = pdev_s + log_dz_tot;
        scalar_t l_last = ok ? A_s[sym_idx(P - 1, P - 1)] : scalar_t(0.0);
        scalar_t se = (ok && l_last > tny) ? (scalar_t(1.0) / l_last) : scalar_t(0.0);
        scalar_t btx = beta_s[P - 1];
        bool valid = ok && isfinite(lap_dev) && isfinite(btx) && isfinite(se) && (se > scalar_t(0.0));
        dev_out[m] = valid ? lap_dev : inf_val<scalar_t>();
        beta_tx_out[m] = btx;
        se_tx_out[m] = se;
        valid_out[m] = valid;
    }
}

}  // namespace

// ============================================================================
// C++ Dispatch Wrappers
// ============================================================================

#define DISPATCH_LAPLACE_T(T_VAL)                                              \
    case T_VAL:                                                                \
        laplace_ar1_eval_kernel<scalar_t, T_VAL><<<M, block_dim>>>(            \
            params.data_ptr<scalar_t>(),                                       \
            idx.data_ptr<int64_t>(),                                           \
            Y_bit.data_ptr<scalar_t>(),                                        \
            X_itp.data_ptr<scalar_t>(),                                        \
            K_it.data_ptr<scalar_t>(),                                         \
            u_in.data_ptr<scalar_t>(),                                         \
            f_out.data_ptr<scalar_t>(),                                        \
            grad_out.data_ptr<scalar_t>(),                                     \
            u_out.data_ptr<scalar_t>(),                                        \
            M, I_c, p_dim, n_inner, inner_tol                                  \
        );                                                                     \
        break

std::vector<torch::Tensor> laplace_ar1_eval_cuda(
    torch::Tensor params,
    torch::Tensor idx,
    torch::Tensor Y_bit,
    torch::Tensor X_itp,
    torch::Tensor K_it,
    torch::Tensor u_in,
    int64_t n_inner,
    double inner_tol
) {
    params = params.contiguous();
    idx = idx.contiguous();
    Y_bit = Y_bit.contiguous();
    X_itp = X_itp.contiguous();
    K_it = K_it.contiguous();
    u_in = u_in.contiguous();

    const int M = static_cast<int>(params.size(0));
    const int D = static_cast<int>(params.size(1));
    const int I_c = static_cast<int>(Y_bit.size(1));
    const int T = static_cast<int>(Y_bit.size(2));
    const int p_dim = D - 3;

    auto opts = params.options();
    auto f_out = torch::empty({M}, opts);
    auto grad_out = torch::empty({M, D}, opts);
    auto u_out = torch::empty({M, I_c, T}, opts);
    if (M == 0) {
        return {f_out, grad_out, u_out};
    }

    int block_dim = 32;
    if (I_c > 96) block_dim = 128;
    else if (I_c > 32) block_dim = 64;

    AT_DISPATCH_FLOATING_TYPES(params.scalar_type(), "laplace_ar1_eval_cuda", [&] {
        switch (T) {
            DISPATCH_LAPLACE_T(2);
            DISPATCH_LAPLACE_T(3);
            DISPATCH_LAPLACE_T(4);
            DISPATCH_LAPLACE_T(5);
            DISPATCH_LAPLACE_T(6);
            DISPATCH_LAPLACE_T(7);
            DISPATCH_LAPLACE_T(8);
            default:
                TORCH_CHECK(false, "laplace_ar1_eval_cuda supports 2 <= Tp <= 8, got ", T);
        }
    });
    return {f_out, grad_out, u_out};
}

#define DISPATCH_CONT_CASE(T_VAL, P_VAL)                                       \
    case ((T_VAL) * 32 + (P_VAL)):                                             \
        cont_ar1_eval_kernel<scalar_t, T_VAL, P_VAL><<<grid, block>>>(         \
            psi_sub.data_ptr<scalar_t>(),                                      \
            idx.data_ptr<int64_t>(),                                           \
            S_YY.data_ptr<scalar_t>(),                                         \
            G_YX.data_ptr<scalar_t>(),                                         \
            M_XX.data_ptr<scalar_t>(),                                         \
            K_t.data_ptr<scalar_t>(),                                          \
            f_out.data_ptr<scalar_t>(),                                        \
            grad_out.data_ptr<scalar_t>(),                                     \
            need_aux ? beta_out.data_ptr<scalar_t>() : nullptr,                \
            need_aux ? ainv_last_out.data_ptr<scalar_t>() : nullptr,           \
            M, I_c, reml, need_aux                                             \
        );                                                                     \
        break

std::vector<torch::Tensor> cont_ar1_eval_cuda(
    torch::Tensor psi_sub,
    torch::Tensor idx,
    torch::Tensor S_YY,
    torch::Tensor G_YX,
    torch::Tensor M_XX,
    torch::Tensor K_t,
    int64_t I_c,
    bool reml,
    bool need_aux
) {
    psi_sub = psi_sub.contiguous();
    idx = idx.contiguous();
    S_YY = S_YY.contiguous();
    G_YX = G_YX.contiguous();
    M_XX = M_XX.contiguous();
    K_t = K_t.contiguous();

    const int M = static_cast<int>(psi_sub.size(0));
    const int T = static_cast<int>(S_YY.size(1));
    const int P = static_cast<int>(G_YX.size(3));

    auto opts = psi_sub.options();
    auto f_out = torch::empty({M}, opts);
    auto grad_out = torch::empty({M, 4}, opts);
    auto beta_out = need_aux ? torch::empty({M, P}, opts) : torch::empty({0}, opts);
    auto ainv_last_out = need_aux ? torch::empty({M}, opts) : torch::empty({0}, opts);
    if (M == 0) {
        return {f_out, grad_out, beta_out, ainv_last_out};
    }

    const int block = 128;
    const int grid = (M + block - 1) / block;
    const int key = T * 32 + P;

    TORCH_CHECK(psi_sub.scalar_type() == torch::kFloat64, "cont_ar1_eval_cuda requires float64");
    using scalar_t = double;
    switch (key) {
        DISPATCH_CONT_CASE(2, 2); DISPATCH_CONT_CASE(2, 3);
        DISPATCH_CONT_CASE(3, 2); DISPATCH_CONT_CASE(3, 4);
        DISPATCH_CONT_CASE(4, 2); DISPATCH_CONT_CASE(4, 5);
        DISPATCH_CONT_CASE(5, 2); DISPATCH_CONT_CASE(5, 6);
        DISPATCH_CONT_CASE(6, 2); DISPATCH_CONT_CASE(6, 7);
        DISPATCH_CONT_CASE(7, 2); DISPATCH_CONT_CASE(7, 8);
        DISPATCH_CONT_CASE(8, 2); DISPATCH_CONT_CASE(8, 9);
        default:
            TORCH_CHECK(false, "cont_ar1_eval_cuda unsupported (Tp, p)=(", T, ",", P, ")");
    }
    return {f_out, grad_out, beta_out, ainv_last_out};
}

#define DISPATCH_PIRLS_CASE(T_VAL, P_VAL)                                      \
    case ((T_VAL) * 32 + (P_VAL)):                                             \
        pirls_binary_kernel<scalar_t, T_VAL, P_VAL><<<M, block_dim>>>(         \
            theta_sub.data_ptr<scalar_t>(),                                    \
            idx.data_ptr<int64_t>(),                                           \
            Y_bit.data_ptr<scalar_t>(),                                        \
            K_it.data_ptr<scalar_t>(),                                         \
            tx_it.data_ptr<scalar_t>(),                                        \
            beta_in.data_ptr<scalar_t>(),                                      \
            z_in.data_ptr<scalar_t>(),                                         \
            dev_out.data_ptr<scalar_t>(),                                      \
            beta_tx_out.data_ptr<scalar_t>(),                                  \
            se_tx_out.data_ptr<scalar_t>(),                                    \
            valid_out.data_ptr<bool>(),                                        \
            beta_out.data_ptr<scalar_t>(),                                     \
            z_out.data_ptr<scalar_t>(),                                        \
            M, I_c, adjust_period, static_cast<int>(max_iter), tol             \
        );                                                                     \
        break

std::vector<torch::Tensor> pirls_binary_cuda(
    torch::Tensor theta_sub,
    torch::Tensor idx,
    torch::Tensor Y_bit,
    torch::Tensor K_it,
    torch::Tensor tx_it,
    torch::Tensor beta_in,
    torch::Tensor z_in,
    bool adjust_period,
    int64_t max_iter,
    double tol
) {
    theta_sub = theta_sub.contiguous();
    idx = idx.contiguous();
    Y_bit = Y_bit.contiguous();
    K_it = K_it.contiguous();
    tx_it = tx_it.contiguous();
    beta_in = beta_in.contiguous();
    z_in = z_in.contiguous();

    const int M = static_cast<int>(theta_sub.size(0));
    const int I_c = static_cast<int>(Y_bit.size(1));
    const int T = static_cast<int>(Y_bit.size(2));
    const int P = static_cast<int>(beta_in.size(1));

    auto opts = theta_sub.options();
    auto dev_out = torch::empty({M}, opts);
    auto beta_tx_out = torch::empty({M}, opts);
    auto se_tx_out = torch::empty({M}, opts);
    auto valid_out = torch::empty({M}, opts.dtype(torch::kBool));
    auto beta_out = torch::empty({M, P}, opts);
    auto z_out = torch::empty({M, I_c}, opts);
    if (M == 0) {
        return {dev_out, beta_tx_out, se_tx_out, valid_out, beta_out, z_out};
    }

    int block_dim = 32;
    if (I_c > 96) block_dim = 128;
    else if (I_c > 32) block_dim = 64;
    if (I_c > block_dim * 8) {
        block_dim = 256;
    }
    TORCH_CHECK(I_c <= block_dim * 8, "pirls_binary_cuda supports I <= 2048");

    const int key = T * 32 + P;
    TORCH_CHECK(theta_sub.scalar_type() == torch::kFloat64, "pirls_binary_cuda requires float64");
    using scalar_t = double;
    switch (key) {
        DISPATCH_PIRLS_CASE(2, 2); DISPATCH_PIRLS_CASE(2, 3);
        DISPATCH_PIRLS_CASE(3, 2); DISPATCH_PIRLS_CASE(3, 4);
        DISPATCH_PIRLS_CASE(4, 2); DISPATCH_PIRLS_CASE(4, 5);
        DISPATCH_PIRLS_CASE(5, 2); DISPATCH_PIRLS_CASE(5, 6);
        DISPATCH_PIRLS_CASE(6, 2); DISPATCH_PIRLS_CASE(6, 7);
        DISPATCH_PIRLS_CASE(7, 2); DISPATCH_PIRLS_CASE(7, 8);
        DISPATCH_PIRLS_CASE(8, 2); DISPATCH_PIRLS_CASE(8, 9);
        default:
            TORCH_CHECK(false, "pirls_binary_cuda unsupported (Tp, p)=(", T, ",", P, ")");
    }
    return {dev_out, beta_tx_out, se_tx_out, valid_out, beta_out, z_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("laplace_ar1_eval_cuda", &laplace_ar1_eval_cuda, "Fused Binary AR(1) Laplace evaluation (CUDA)");
    m.def("cont_ar1_eval_cuda", &cont_ar1_eval_cuda, "Fused Continuous AR(1) REML/ML evaluation (CUDA)");
    m.def("pirls_binary_cuda", &pirls_binary_cuda, "Fused Binary RI PIRLS solver (CUDA)");
}
