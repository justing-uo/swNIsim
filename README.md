# swNIsim 0.2.0 (GPU-Accelerated Edition)

`swNIsim` estimates power for non-inferiority tests in repeated cross-sectional stepped-wedge cluster randomized trials with binary or continuous outcomes.

## What's New in 0.2.0 (GPU Support for Google Colab)

- **Batched GPU & PyTorch acceleration (`backend = "auto"` / `"gpu"`)**: Evaluates tens of thousands of mixed-model replicates simultaneously in `float64` using exact Sherman-Morrison (`lmer`, ML & REML with cell weights `K`) and Schur-complement PIRLS (`glmer` binomial, `nAGQ = 0`) solvers.
- **Zero-compile Google Colab setup**: Uses Google Colab's pre-installed CUDA PyTorch via `reticulate` (when installed) or an automatic zero-dependency Python bridge—no need to wait 5–10 minutes compiling `lme4` or downloading `libtorch` on fresh Colab sessions.
- **100% backward compatible**: Retains the exact `lme4` reference engine (`backend = "lme4"`) and all existing `sw_ni_power()`, `sw_ni_grid()`, `sw_ni_design()`, Shiny, and Word export interfaces.

## Quickstart in Google Colab

### A. Google Colab with **R Runtime** (`Runtime -> Change runtime type -> R` + `T4 GPU`)

Upload `swNIsim_0.2.0.tar.gz` to the Colab session files pane (left sidebar), then run:

```r
# Optional: install reticulate for in-process Python calls (works even without it!)
# install.packages("reticulate")

install.packages("swNIsim_0.2.0.tar.gz", repos = NULL, type = "source")
library(swNIsim)

# Verify T4 / L4 / A100 GPU detection
sw_ni_gpu_status()

# Run 100,000 continuous trials on GPU (< 1 second on T4)
ans_cont <- sw_ni_power(
  outcome = "continuous",
  trials = 100000,
  Tp = 9,
  no_clusters_per_step = 2,
  K = 5,
  mu = 21,
  sdw = 10,
  sdc = 4.2,
  margin = 4.2,
  REML = TRUE,
  backend = "gpu",
  return_details = TRUE
)
ans_cont$power
ans_cont$device
```

### B. Google Colab with **Default Python Runtime** (`Runtime -> Change runtime type -> Python 3` + `T4 GPU`)

In a Colab Python notebook cell, upload `swNIsim_0.2.0.tar.gz` and `swNIsim_worked_examples_gpu_colab.R`, then run:

```python
!R CMD INSTALL swNIsim_0.2.0.tar.gz
!Rscript swNIsim_worked_examples_gpu_colab.R 100000
```

Or interactively across cells with `rpy2`:

```python
%load_ext rpy2.ipython
```
```r
%%R
install.packages("swNIsim_0.2.0.tar.gz", repos = NULL, type = "source")
library(swNIsim)
print(sw_ni_gpu_status())
```

## Choosing a Simulation Backend

`sw_ni_power()` and `sw_ni_grid()` accept `backend`:

- `"auto"` *(default)*: Uses PyTorch GPU (`cuda` / `mps`) if available, else batched PyTorch CPU if available, else CPU `lme4`.
- `"gpu"`: Forces GPU execution via PyTorch (errors immediately if no GPU is enabled).
- `"torch_cpu"`: Runs the batched PyTorch solver on CPU.
- `"lme4"`: Runs the original per-trial `lme4::lmer` / `lme4::glmer` CPU loop (requires `install.packages("lme4")`).

You can also tune `batch_size` (e.g. `batch_size = 10000L`) to control peak GPU VRAM usage, or pin a specific Python interpreter before loading via `Sys.setenv(SWNISIM_PYTHON = "/path/to/python3")`.

## Unequal Clusters, Secular Period Effects, and Matrix `K`

All design features work identically across `"gpu"`, `"torch_cpu"`, and `"lme4"`:

```r
d <- sw_ni_design(Tp = 5, clusters_per_sequence = c(3, 4, 5, 4))
K_matrix <- matrix(20L, nrow = d$I, ncol = d$Tp)
K_matrix[, 5] <- 30L

ans <- sw_ni_power(
  outcome = "binary",
  trials = 10000,
  Tp = 5,
  clusters_per_sequence = c(3, 4, 5, 4),
  K = K_matrix,
  period_effects = c(0, 0.10, 0.15, 0.20, 0.20),
  b1 = 0.5,
  sdc = 1,
  theta = 0,
  margin = 0.25,
  backend = "auto",
  return_details = TRUE
)
```

## Statistical Model

For continuous outcomes, cluster-period means are generated with variance `sdw^2 / K_ij` and analyzed using:

```r
lme4::lmer(mu_ij ~ period + delayed_tx + (1 | cluster), weights = K)
```

For binary outcomes, aggregate binomial counts are analyzed using:

```r
lme4::glmer(
  cbind(success, fail) ~ period + delayed_tx + (1 | cluster),
  family = binomial(), nAGQ = 0
)
```

The one-sided non-inferiority test rejects when

```r
1 - pnorm((estimate + margin) / standard_error) < sig.level
```
