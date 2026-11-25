# swNIsim 0.2.0

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
