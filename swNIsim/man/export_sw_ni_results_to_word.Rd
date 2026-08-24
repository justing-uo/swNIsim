\name{export_sw_ni_results_to_word}
\alias{export_sw_ni_results_to_word}
\title{Export simulation results to Word}
\usage{export_sw_ni_results_to_word(results, target = "sw_ni_results.docx", title = "Stepped-Wedge Non-Inferiority Results")}
\arguments{
  \item{results}{A data frame of simulation results.}
  \item{target}{Output docx path.}
  \item{title}{Document title.}
}
\value{Invisibly returns \code{target}.}
\description{Writes a simulation result table to a Word document.}
