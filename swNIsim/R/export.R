#' Export simulation results to a Word document
#'
#' Writes a power-simulation results table to a `.docx` file using `officer` and
#' `flextable`.
#'
#' @param results A data frame of simulation results, usually from [sw_ni_grid()].
#' @param target Output `.docx` path.
#' @param title Document title.
#'
#' @return Invisibly returns `target`.
#' @export
export_sw_ni_results_to_word <- function(results,
                                         target = "sw_ni_results.docx",
                                         title = "Stepped-Wedge Non-Inferiority Results") {
  if (!requireNamespace("officer", quietly = TRUE)) {
    stop("Package 'officer' is required for Word export. Install it with install.packages('officer').", call. = FALSE)
  }
  if (!requireNamespace("flextable", quietly = TRUE)) {
    stop("Package 'flextable' is required for Word export. Install it with install.packages('flextable').", call. = FALSE)
  }

  if (!is.data.frame(results)) {
    stop("results must be a data frame.", call. = FALSE)
  }

  ft <- flextable::flextable(results)
  ft <- flextable::autofit(ft)

  doc <- officer::read_docx()
  doc <- officer::body_add_par(doc, title, style = "heading 1")
  doc <- flextable::body_add_flextable(doc, ft)

  print(doc, target = target)
  invisible(target)
}
