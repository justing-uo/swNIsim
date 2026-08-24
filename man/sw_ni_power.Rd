\name{sw_ni_power}
\alias{sw_ni_power}
\title{Estimate power for a stepped-wedge non-inferiority design}
\usage{
sw_ni_power(outcome = c("binary", "continuous"), seed = 2026,
  trials = 5000, Tp = 5, no_clusters_per_step = 4,
  clusters_per_sequence = NULL, K = 20, b1 = 0.5, mu = 5,
  sdw = 2, sdc = 1, theta = 0, margin = NULL, delay = NULL,
  period_effects = NULL, adjust_period = TRUE, sig.level = 0.025,
  REML = TRUE, failure_handling = c("nonrejection", "exclude", "error"),
  parallel = FALSE,
  ncores = .default_ncores(),
  return_details = FALSE)
}
\arguments{
  \item{outcome}{Either \code{"binary"} or \code{"continuous"}.}
  \item{seed}{Random-number seed.}
  \item{trials}{Number of simulated trials.}
  \item{Tp}{Number of time periods.}
  \item{no_clusters_per_step}{One cluster count for a balanced design or a vector of length \code{Tp - 1}.}
  \item{clusters_per_sequence}{Optional vector of cluster counts by crossover sequence.}
  \item{K}{A scalar, length-\code{I} vector, length-\code{n_cells} vector, or \code{I} by \code{Tp} matrix of cluster-period sample sizes.}
  \item{b1}{Binary baseline log-odds intercept.}
  \item{mu}{Continuous baseline mean.}
  \item{sdw}{Continuous individual-level within-cluster standard deviation.}
  \item{sdc}{Cluster random-intercept standard deviation.}
  \item{theta}{True treatment effect.}
  \item{margin}{Non-inferiority margin on the model scale.}
  \item{delay}{Optional post-crossover exposure vector of length \code{Tp - 1}.}
  \item{period_effects}{Optional secular effects of length \code{Tp}, or \code{Tp - 1} with period 1 as reference.}
  \item{adjust_period}{Include categorical period fixed effects in the fitted model.}
  \item{sig.level}{One-sided significance level.}
  \item{REML}{Use REML for the continuous LMM.}
  \item{failure_handling}{How to handle failed fits: count as non-rejections, exclude, or stop.}
  \item{parallel}{Parallelize simulation replicates.}
  \item{ncores}{Number of workers.}
  \item{return_details}{Return Monte Carlo SE, failures, design totals, and normalized parameters.}
}
\value{Numeric power or a detailed list.}
\description{Main binary/continuous simulation interface with period adjustment, unequal sequence sizes, and unequal cluster-period sample sizes.}
