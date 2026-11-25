# swNIsim 1.3

Combines the 1.2 line with the 0.3.0 GPU edition, which was developed in
parallel from 0.1.x.

- From 0.3.0: batched PyTorch GPU/CPU engine, `backend`, `batch_size` and
  `device` arguments, `sw_ni_gpu_status()`, Google Colab support, and fused CUDA
  kernels. `lme4` moves to Suggests so GPU-only installs do not need it.
- From 0.3.0: stationary AR(1) cluster-by-period random effects. The new `rho`
  argument is the autocorrelation of the cluster-by-period effect (`sdcp`)
  across periods. `rho = 0` (default) is the 1.2 nested exchangeable
  structure; a non-zero `rho` runs on PyTorch, which fits the AR(1) model and
  requires equal cluster-period sizes.
- From 1.1 and 1.2: `build_margin()`, `sw_ni_sample_size()`, `fit_cp`, `nAGQ`,
  nested exchangeable correlation, and the `correlation` output, which now also
  reports AR(1) structures.
- `backend = "auto"` uses PyTorch when it is available, except for settings
  only `lme4` supports (`nAGQ != 0`, an explicit `fit_cp`, or `sdcp > 0` with
  `rho = 0`), which use `lme4` as in 1.2. Use `backend = "lme4"` to reproduce
  1.2 results seed-for-seed.
- `rho` is available in `sw_ni_grid()`, `build_sw_ni_grid()`,
  `sw_ni_sample_size()` and the Shiny app.

# swNIsim 1.2

- New nested exchangeable correlation option. `sdcp` adds an independent
  cluster-by-period random effect to the data-generating model for both
  outcomes, so the within-period and between-period ICCs can differ. The
  default, `sdcp = 0`, gives the exchangeable structure and reproduces 1.1
  exactly for the same seed.
- New `fit_cp` argument for binary outcomes: whether the fitted model includes
  `(1 | cluster:period)`. By default it is included when `sdcp > 0`; set
  `FALSE` to evaluate an analysis that wrongly assumes exchangeable
  correlation.
- For continuous outcomes the analysis of cluster-period means is unchanged; with
  a common cluster-period size it absorbs the cluster-by-period effect in its
  residual variance. A warning is given when `sdcp > 0` and cluster-period sizes
  differ, because the precision weights are then approximate.
- `return_details = TRUE` now also returns `correlation`: the within-period ICC,
  between-period ICC and cluster autocorrelation implied by the inputs.
- `sdcp` and `fit_cp` are available in `sw_ni_grid()`, `build_sw_ni_grid()`
  (`sdcp`), `sw_ni_sample_size()` and the Shiny app.
- Exponential decay of the between-period correlation is not yet implemented.

# swNIsim 1.1

- `build_margin()` now requires an explicit `method`: `"preserve_control"`,
  `"preserve_benefit"`, `"threshold"` or `"direct"`. It no longer supplies
  default reference values or preserved fractions, rejects arguments the chosen
  method does not use, and validates input ranges. New `p_min` and `mu_min`
  arguments accept a clinically specified minimum directly. The result also
  reports the method, the reference value and the minimum acceptable value.
  Calls written for 1.0 must now name a method.
- New `sw_ni_sample_size()` finds the smallest common cluster-period size `K`,
  or number of clusters per sequence, whose simulated power meets a target. It
  bisects between bounds or evaluates a supplied set of candidates, and flags
  answers within two Monte Carlo SEs of the target.
- New `nAGQ` argument for binary outcomes in `sw_ni_power()`, `sw_ni_grid()`,
  `build_sw_ni_grid()` and the Shiny app. The default, 0, is unchanged.
- `sw_ni_power()`, `sw_ni_grid()`, `sw_ni_sample_size()` and the Shiny app now
  warn when `adjust_period = FALSE` (one warning per grid or search).
- The one-sided Wald test is unchanged, and results under default settings are
  identical to 1.0 for the same seed.
- Help pages are now generated from the source with roxygen2.

# swNIsim 1.0

- Added `build_margin()`.

# swNIsim 0.3.0 / 0.2.0 (GPU edition, merged in 1.3)

- Added batched PyTorch GPU and CPU simulation engine (`inst/python/swnisim_torch.py`)
  for both continuous (`lmer` ML/REML with cell weights `K`) and binary (`glmer`
  binomial with `nAGQ = 0`) outcomes using exact Sherman-Morrison and Schur-complement
  mixed-model formulations in `float64`.
- Added `backend = c("auto", "gpu", "torch_cpu", "lme4")`, `batch_size`, and
  `device` arguments to `sw_ni_power()` and `sw_ni_grid()`.
- Added automatic Python/`torch` discovery supporting both in-process `reticulate`
  and a zero-dependency Python CLI fallback so Google Colab sessions work out of
  the box without requiring C++ package compilation.
- Added `sw_ni_gpu_status()` diagnostic helper and backend selector in `run_sw_ni_app()`.
- Moved `lme4` from hard `Imports` to `Suggests` so GPU-only Google Colab sessions
  can install `swNIsim` in seconds without compiling `lme4` from source.

# swNIsim 0.1.3

- Capped automatically created PSOCK clusters at 64 workers to stay below R's
  default 128-connection limit on high-core Linux servers.
- Honored common scheduler CPU allocations when selecting the default worker
  count.
- Added an optional reusable `cluster` argument to `sw_ni_power()` and
  `sw_ni_grid()`.
- Improved cluster cleanup and errors when worker startup fails.

# swNIsim 0.1.2

- Added fixed categorical period adjustment to binary and continuous analysis models.
- Added user-specified secular `period_effects` to both data-generating models.
- Added unequal `clusters_per_sequence` support.
- Added unequal `K` support at cluster or cluster-period level, including matrices.
- Added precision weights for continuous cluster-period means when `K` varies.
- Added conservative failed-fit handling and Monte Carlo standard errors.
- Added actual total clusters, cluster-period cells, and observations to detailed output.
- Extended the Shiny app and grid runner for the new arguments.
