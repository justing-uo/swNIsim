# swNIsim 1.3

`swNIsim` estimates power for non-inferiority tests in repeated cross-sectional stepped-wedge cluster randomized trials with binary or continuous outcomes.

## What changed in 1.3

Version 1.3 combines the 1.2 line with the 0.3.0 GPU edition, which was developed in parallel:

- **GPU and batched CPU acceleration.** A PyTorch engine runs tens of thousands of replicates at once on a CUDA/MPS GPU or CPU. `backend`, `batch_size` and `device` choose and tune it; `sw_ni_gpu_status()` reports what was found. It runs on Google Colab without compiling `lme4`.
- **AR(1) cluster-by-period effects.** `rho` sets the autocorrelation of the cluster-by-period effect across periods, so the between-period correlation decays with distance. `rho = 0` (default) is the 1.2 nested exchangeable structure.
- Everything from 1.1 and 1.2 is kept: `build_margin()`, `sw_ni_sample_size()`, `fit_cp`, `nAGQ` and nested exchangeable correlation.
- Use `backend = "lme4"` to reproduce 1.2 results seed-for-seed. See [Choosing a simulation backend](#choosing-a-simulation-backend).

## What changed in 1.2

- Nested exchangeable correlation: `sdcp` adds a cluster-by-period random effect, so the within-period and between-period ICCs can differ.
- For binary outcomes, `fit_cp` controls whether the analysis includes the cluster-by-period effect, so correctly specified and misspecified analyses can both be evaluated.
- With `sdcp = 0` (the default), results are identical to 1.1 and 1.0 for the same seed.

## What changed in 1.1

- `build_margin()` requires an explicit `method` and has no default reference values or preserved fractions.
- `sw_ni_sample_size()` finds the smallest cluster-period size or number of clusters meeting a power target.
- Binary analyses accept an `nAGQ` argument; the default (0) is unchanged.
- `adjust_period = FALSE` now gives a warning.

The package supports exchangeable, nested exchangeable and AR(1) cluster-by-period correlation for repeated cross-sectional designs. It does not yet model closed-cohort individual effects, missing outcomes, cluster dropout, or heterogeneous treatment effects.

## Installation

```r
library(devtools)
install_github("justing-uo/swNIsim") # Or install.packages("swNIsim_1.3.tar.gz", repos = NULL, type = "source")
library(swNIsim)
```

For GPU or batched CPU acceleration, install Python (>= 3.9) with `torch`; `reticulate` is optional and makes the calls in-process. Set `Sys.setenv(SWNISIM_PYTHON = "/path/to/python3")` to pick a specific interpreter. Check what was found with:

```r
sw_ni_gpu_status()
```

## Choosing a simulation backend

`sw_ni_power()`, `sw_ni_grid()` and `sw_ni_sample_size()` accept `backend`:

- `"auto"` *(default)*: PyTorch on a GPU (`cuda` / `mps`) if available, else batched PyTorch on CPU, else `lme4`. Settings only `lme4` supports (`nAGQ != 0`, an explicit `fit_cp`, or nested exchangeable correlation, `sdcp > 0` with `rho = 0`) always use `lme4`.
- `"gpu"`: PyTorch on a GPU; errors if none is found.
- `"torch_cpu"`: batched PyTorch on CPU.
- `"lme4"`: the per-trial `lme4::lmer` / `lme4::glmer` reference engine. Use it to reproduce results from 1.2 and earlier seed-for-seed.

The engines fit the same models but draw random numbers differently, so the same seed gives slightly different power estimates on each. `batch_size` (e.g. `10000L`) controls peak GPU memory.

| Setting | `lme4` | PyTorch |
|---|---|---|
| Exchangeable (`sdcp = 0`) | yes | yes |
| Nested exchangeable (`sdcp > 0`, `rho = 0`) | yes (default under `"auto"`) | fitted as AR(1) |
| AR(1) (`sdcp > 0`, `rho != 0`) | no | yes, equal `K` only |
| `fit_cp`, `nAGQ != 0` | yes | no |
| Unequal `K` | yes | yes (`sdcp = 0`) |

## Google Colab quickstart

Choose `Runtime -> Change runtime type -> R` and a GPU (e.g. T4). Upload the tarball to the session's files pane, then run:

```r
library(swNIsim)
sw_ni_gpu_status()

ans <- sw_ni_power(
  outcome = "continuous", trials = 100000,
  Tp = 9, no_clusters_per_step = 2, K = 5,
  mu = 21, sdw = 10, sdc = 4.2, margin = 4.2,
  backend = "gpu", return_details = TRUE
)
ans$power
ans$device
```

In a Python runtime, `!R CMD INSTALL swNIsim_1.3.tar.gz` installs it, and `%load_ext rpy2.ipython` followed by `%%R` cells runs R code. A longer script is in [`inst/colab/colab_quickstart.R`](inst/colab/colab_quickstart.R).

## Converting a clinical margin

Larger outcome values are favourable. For a binary outcome, the margin is a cluster-conditional log odds ratio.

```r
# Keep at least 80% of a 30% control success rate (minimum acceptable 24%)
m <- build_margin("binary", method = "preserve_control", p1 = 0.30, f1 = 0.80)
m$margin  # 0.305
m$b1      # qlogis(0.30)

# Continuous: a 20% loss from a control mean of 21 days
build_margin("continuous", method = "preserve_control", mu = 21, f1 = 0.80)$margin  # 4.2
```

Other methods are `"preserve_benefit"` (keep a fraction `f2` of the control's benefit over a benchmark `p0` or `mu0`), `"threshold"` (give `p_min` or `mu_min` directly) and `"direct"` (give `margin` on the model scale).

## Nested exchangeable correlation

```r
# Continuous
ans <- sw_ni_power(
  outcome = "continuous", trials = 10000, Tp = 7, no_clusters_per_step = 10, K = 20,
  mu = 5, sdw = 2, sdc = 1, sdcp = 0, theta = 0, margin = 0.15,
  return_details = TRUE
)
ans$power

# Binary
sw_ni_power(
  outcome = "binary", trials = 10000, Tp = 5, no_clusters_per_step = 10, K = 30,
  b1 = 0.5, sdc = 1, sdcp = 0, theta = 0, margin = 0.18,
)
ans$power
```

## AR(1) cluster-by-period effects

With `sdc = 0` and `rho != 0`, the cluster-by-period effects follow a stationary AR(1) process, so the correlation between two periods in the same cluster decays with the distance between them. These designs run on PyTorch, which fits the AR(1) model (`glmmTMB`-style; Laplace approximation for binary outcomes). Cluster-period sizes must be equal.

```r
ans <- sw_ni_power(
  outcome = "continuous", trials = 10000, Tp = 7, no_clusters_per_step = 4, K = 20,
  mu = 5, sdw = 2, sdc = 0, sdcp = 1, rho = 0.6, theta = 0, margin = 0.15,
  backend = "auto", return_details = TRUE
)
ans$power
```

## Finding the smallest design that meets a power target

```r
res <- sw_ni_sample_size(
  target_power = 0.90, search = "K", lower = 2, upper = 20,
  outcome = "continuous", Tp = 9, no_clusters_per_step = 2,
  mu = 21, sdw = 10, sdc = 4.2, theta = 0, margin = 4.2,
  trials = 10000
)
res            # smallest K, its power and Monte Carlo SE, and the value below it
res$evaluated  # every candidate simulated
```

Use `search = "clusters"` to search over the number of clusters per sequence instead, or `values = 2:10` to evaluate every candidate and obtain the power curve. If the answer is flagged as within two Monte Carlo SEs of the target, confirm it with more trials.

## Worked examples from real trials

Three planning examples based on published stepped-wedge trials. All use a one-sided test at `sig.level = 0.025`, true effect `theta = 0`, categorical period adjustment with no secular trend, and failed fits counted as non-rejections.

| Example | Outcome | Periods (`Tp`) | Clusters per step | Inputs | Margin |
|---|---|---|---|---|---|
| TheraPPP overall | Continuous | 9 | 2 | `mu = 21`, `sdw = 10`, `sdc = 4.2` | 4.2 (20% of the control mean) |
| TheraPPP ARDS subgroup | Continuous | 9 | 2 | `mu = 11`, `sdw = 10`, `sdc = 1` | 2.2 (20% of the control mean) |
| PRIM-ER | Binary (non-admission is favourable) | 31 | 1 | control rate 30%, ICC 0.10 | 30% → 24%, log OR 0.305 |

For PRIM-ER, `b1 = qlogis(0.30)` and `sdc = sqrt((0.10 / 0.90) * pi^2 / 3)`; probabilities are conditional on a cluster random intercept of zero.

Power from 100,000 simulated trials per candidate `K` (cluster-period size):

| Example | Target | K | Total N | Power | MCSE |
|---|---|---|---|---|---|
| TheraPPP overall | 0.90 | 4 | 576 | 0.855 | 0.0011 |
| | | **5** | **720** | **0.916** | 0.0009 |
| | | 6 | 864 | 0.954 | 0.0007 |
| TheraPPP ARDS | 0.90 | 13 | 1,872 | 0.888 | 0.0010 |
| | | **14** | **2,016** | **0.905** | 0.0009 |
| PRIM-ER | 0.85 | 5 | 4,650 | 0.793 | 0.0013 |
| | | **6** | **5,580** | **0.859** | 0.0011 |
| | | 7 | 6,510 | 0.902 | 0.0009 |

Bold rows are the smallest `K` meeting each target. No estimate is within two Monte Carlo SEs of its target.

The script and its full output are in `inst/examples/`:

- [`swNIsim_worked_examples_v1.0.R`](inst/examples/swNIsim_worked_examples_v1.0.R)
- [`swNIsim_worked_examples_v1.0_100000_reps.csv`](inst/examples/swNIsim_worked_examples_v1.0_100000_reps.csv)

```sh
Rscript swNIsim_worked_examples_v1.0.R          # 100,000 trials per candidate
Rscript swNIsim_worked_examples_v1.0.R 10000    # quicker pilot run
```

The script checks that swNIsim 1.0 is installed, since that version produced the manuscript's numbers. With `backend = "lme4"`, 1.3 uses the same seed sequence and gives matching estimates; exact rejection counts can differ by a few in 100,000 across `lme4` versions (e.g. 91,633 vs. 91,639 for TheraPPP overall, K = 5). A single example can also be run directly in 1.3:

```r
sw_ni_power(
  outcome = "continuous", seed = 20261111, trials = 100000,
  Tp = 9, no_clusters_per_step = 2, K = 5,
  mu = 21, sdw = 10, sdc = 4.2, theta = 0, margin = 4.2,
  sig.level = 0.025, backend = "lme4"
)  # TheraPPP overall, 0.916
```

## Simple continuous example

```r
sw_ni_power(
  outcome = "continuous",
  trials = 100,
  Tp = 5,
  no_clusters_per_step = 4,
  K = 20,
  mu = 5,
  sdw = 2,
  sdc = 1,
  theta = 0,
  margin = 0.15
)
```

Categorical period adjustment is used by default even when the true period effects are zero. Setting `adjust_period = FALSE` gives a warning: in a stepped-wedge trial, exposure is confounded with calendar time.

## Unequal clusters and secular period effects

```r
ans <- sw_ni_power(
  outcome = "continuous",
  seed = 2026,
  trials = 1000,
  Tp = 5,
  clusters_per_sequence = c(3, 4, 5, 4),
  K = 20,
  period_effects = c(0, 0.10, 0.15, 0.20, 0.20),
  adjust_period = TRUE,
  mu = 5,
  sdw = 2,
  sdc = 1,
  theta = 0,
  margin = 0.15,
  return_details = TRUE
)

ans$power
ans$mcse
ans$design
```

## Unequal cluster-period sample sizes

`K` may be:

1. one scalar;
2. a vector of length `I`, constant within each cluster;
3. a vector of length `n_cells`, one size per cluster-period in design order; or
4. an `I` by `Tp` matrix.

```r
d <- sw_ni_design(Tp = 5, clusters_per_sequence = c(3, 4, 5, 4))
K_matrix <- matrix(20L, nrow = d$I, ncol = d$Tp)
K_matrix[, 5] <- 30L

ans <- sw_ni_power(
  outcome = "binary",
  trials = 1000,
  Tp = 5,
  clusters_per_sequence = c(3, 4, 5, 4),
  K = K_matrix,
  b1 = 0.5,
  sdc = 1,
  theta = 0,
  margin = 0.25,
  return_details = TRUE
)
```

## Advanced grid with list-columns

```r
params <- data.frame(
  outcome = c("continuous", "continuous"),
  sdc = c(1, 1.3),
  stringsAsFactors = FALSE
)
params$clusters_per_sequence <- I(list(c(3, 4, 5, 4), c(4, 4, 4, 4)))
params$period_effects <- I(list(c(0, .1, .1, .2, .2), rep(0, 5)))
params$K <- I(list(20L, c(rep(20L, 8), rep(25L, 8))))

res <- sw_ni_grid(
  params,
  trials = 1000,
  Tp = 5,
  mu = 5,
  sdw = 2,
  margin = 0.15,
  parallel = TRUE,
  ncores = 2
)
```

## Interactive interface

```r
install.packages("shiny")
run_sw_ni_app()
```

## Statistical model

For continuous outcomes, cluster-period means are generated with variance `sdw^2 / K_ij` and analyzed using:

```r
lme4::lmer(mu_ij ~ period + delayed_tx + (1 | cluster), weights = K)
```

For binary outcomes, aggregate binomial counts are analyzed using:

```r
lme4::glmer(
  cbind(success, fail) ~ period + delayed_tx + (1 | cluster),  # + (1 | cluster:period) when fit_cp = TRUE
  family = binomial(), nAGQ = 0
)
```

`nAGQ = 0` is the default; set `nAGQ = 1` in `sw_ni_power()` for the Laplace approximation. The PyTorch engine solves the same models in batches (`float64`, `nAGQ = 0` for binary outcomes).

With AR(1) cluster-by-period effects (`rho != 0`), the PyTorch engine fits

```r
glmmTMB::glmmTMB(y ~ period + delayed_tx + (1 | cluster) + ar1(period + 0 | cluster), ...)
```

with REML or ML for continuous outcomes and the Laplace approximation for binary outcomes.

The one-sided non-inferiority test rejects when

```r
1 - pnorm((estimate + margin) / standard_error) < sig.level
```
