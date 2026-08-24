\name{sw_ni_design}
\alias{sw_ni_design}
\title{Build a stepped-wedge design object}
\usage{
sw_ni_design(Tp = 5, no_clusters_per_step = 4,
  clusters_per_sequence = NULL, delay = NULL)
}
\arguments{
  \item{Tp}{Number of time periods. The design has \code{Tp - 1} crossover sequences.}
  \item{no_clusters_per_step}{A positive integer for a balanced design or a vector of length \code{Tp - 1}.}
  \item{clusters_per_sequence}{Optional positive integer vector of length \code{Tp - 1}; takes precedence over \code{no_clusters_per_step}.}
  \item{delay}{Optional numeric vector of length \code{Tp - 1} defining treatment exposure by post-crossover period.}
}
\value{An object of class \code{sw_ni_design} containing the cluster-period data and design totals.}
\description{Builds a repeated cross-sectional cluster-by-period stepped-wedge design and supports unequal numbers of clusters across sequences.}
