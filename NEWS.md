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

# swNIsim 0.1.2

- Added fixed categorical period adjustment to binary and continuous analysis models.
- Added user-specified secular `period_effects` to both data-generating models.
- Added unequal `clusters_per_sequence` support.
- Added unequal `K` support at cluster or cluster-period level, including matrices.
- Added precision weights for continuous cluster-period means when `K` varies.
- Added conservative failed-fit handling and Monte Carlo standard errors.
- Added actual total clusters, cluster-period cells, and observations to detailed output.
- Extended the Shiny app and grid runner for the new arguments.
