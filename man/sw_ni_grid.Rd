\name{sw_ni_grid}
\alias{sw_ni_grid}
\title{Run a parameter grid of stepped-wedge simulations}
\usage{
sw_ni_grid(params, outcome = c("binary", "continuous"), seed = 2026,
  trials = 5000, Tp = 5, no_clusters_per_step = 4,
  clusters_per_sequence = NULL, K = 20, b1 = 0.5, mu = 5,
  sdw = 2, sdc = 1, theta = 0, margin = NULL, delay = NULL,
  period_effects = NULL, adjust_period = TRUE, sig.level = 0.025,
  REML = TRUE, failure_handling = "nonrejection", parallel = TRUE,
  ncores = .default_ncores(),
  return_details = TRUE)
}
\arguments{
  \item{params}{A data frame. Advanced vector parameters may be list-columns.}
  \item{outcome}{Either \code{"binary"} or \code{"continuous"}.}
  \item{seed}{Base seed; row \code{i} uses \code{seed + i - 1}.}
  \item{trials}{Number of simulated trials per scenario.}
  \item{Tp, no_clusters_per_step, clusters_per_sequence, K, b1, mu, sdw, sdc, theta, margin, delay, period_effects, adjust_period, sig.level, REML, failure_handling}{Defaults used when corresponding columns are absent.}
  \item{parallel}{Parallelize across grid rows.}
  \item{ncores}{Number of workers.}
  \item{return_details}{Append Monte Carlo SE, failure information, and design totals.}
}
\value{A data frame containing input parameters and estimated power.}
\description{Runs \code{sw_ni_power()} over every row of a parameter grid.}
