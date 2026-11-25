\name{sw_ni_gpu_status}
\alias{sw_ni_gpu_status}
\title{Check GPU and PyTorch backend availability}
\usage{
sw_ni_gpu_status(refresh = FALSE)
}
\arguments{
  \item{refresh}{Logical; if \code{TRUE}, re-probe Python environments instead of using cached session status.}
}
\value{An object of class \code{sw_ni_gpu_status} summarizing PyTorch, CUDA/MPS GPU, bridge (\code{reticulate} or \code{python_cli}), and \code{lme4} availability.}
\description{Reports available GPU and CPU simulation engines in the current R session, including Google Colab GPU runtimes.}
