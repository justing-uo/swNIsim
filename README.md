# swNIsim 0.1.2

`swNIsim` estimates power for non-inferiority tests in repeated cross-sectional stepped-wedge cluster randomized trials with binary or continuous outcomes.

## What changed in 0.1.2

- Fixed categorical period effects are included in the fitted model by default.
- Users can specify secular period effects in the data-generating model.
- Numbers of clusters may differ across crossover sequences.
- Cluster-period sample sizes may be unequal.
- Continuous models use precision weights when `K` varies.
- Failed fits are counted as non-rejections by default; alternative handling is available.
- Detailed output includes Monte Carlo SE and the actual total number of observations.
- The Shiny interface exposes the new vector inputs.

The package still uses a shared cluster random intercept. It does not yet model cluster-by-period random effects, decaying temporal correlation, closed-cohort individual effects, missing outcomes, cluster dropout, or heterogeneous treatment effects.

## Installation

```r
install.packages("lme4")
install.packages("swNIsim_0.1.2.tar.gz", repos = NULL, type = "source")
library(swNIsim)
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

Categorical period adjustment is used by default even when the true period effects are zero.

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
  cbind(success, fail) ~ period + delayed_tx + (1 | cluster),
  family = binomial(), nAGQ = 0
)
```

The one-sided non-inferiority test rejects when

```r
1 - pnorm((estimate + margin) / standard_error) < sig.level
```
