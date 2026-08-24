\name{build_sw_ni_grid}
\alias{build_sw_ni_grid}
\title{Build a balanced parameter grid}
\usage{
build_sw_ni_grid(outcome = c("binary", "continuous"), Tp = 5,
  no_clusters_per_step = 4, K = 20, sdc = 1, theta = 0,
  margin = NULL, sig.level = 0.025, b1 = 0.5, mu = 5,
  sdw = 2, adjust_period = TRUE, REML = TRUE,
  failure_handling = "nonrejection")
}
\arguments{
  \item{outcome}{Either \code{"binary"} or \code{"continuous"}.}
  \item{Tp, no_clusters_per_step, K, sdc, theta, margin, sig.level}{Values crossed into the grid.}
  \item{b1}{Binary baseline log-odds values.}
  \item{mu, sdw}{Continuous parameter values.}
  \item{adjust_period}{Logical values for categorical period adjustment.}
  \item{REML}{Continuous REML values.}
  \item{failure_handling}{Failed-fit handling values.}
}
\value{A data frame with one row per scenario.}
\description{Creates a conventional scalar grid. Advanced vector inputs can be supplied to \code{sw_ni_grid()} as list-columns.}
