#' Build a stepped-wedge design object
#'
#' Builds the repeated cross-sectional cluster-by-period design used by the
#' binary and continuous non-inferiority simulation engines. Version 0.1.2
#' supports unequal numbers of clusters across crossover sequences.
#'
#' @param Tp Number of time periods. Must be an integer >= 2. The design has
#'   `Tp - 1` crossover sequences.
#' @param no_clusters_per_step Backward-compatible cluster-count argument. It
#'   may be one positive integer for a balanced design or a vector of length
#'   `Tp - 1` for unequal numbers of clusters across sequences.
#' @param clusters_per_sequence Optional positive integer vector of length
#'   `Tp - 1`. When supplied, it takes precedence over
#'   `no_clusters_per_step`.
#' @param delay Optional numeric vector with length `Tp - 1`. Element `r`
#'   specifies the treatment-exposure value at the `r`th post-crossover period.
#'   If `NULL`, every post-crossover period has exposure value 1.
#'
#' @return An object of class `sw_ni_design` containing the cluster-period data,
#'   number of clusters, number of cluster-period cells, and sequence sizes.
#' @export
sw_ni_design <- function(Tp = 5,
                         no_clusters_per_step = 4,
                         clusters_per_sequence = NULL,
                         delay = NULL) {
  .check_scalar_numeric(Tp, "Tp", lower = 2, integer = TRUE)
  Tp <- as.integer(Tp)

  clusters_per_sequence <- .resolve_clusters_per_sequence(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    clusters_per_sequence = clusters_per_sequence
  )
  delay <- .normalize_delay(delay, Tp)

  rows <- vector("list", sum(clusters_per_sequence))
  cluster_id <- 0L
  row_id <- 0L

  for (sequence in seq_len(Tp - 1L)) {
    for (cluster_within_sequence in seq_len(clusters_per_sequence[sequence])) {
      cluster_id <- cluster_id + 1L
      row_id <- row_id + 1L
      time <- seq_len(Tp)
      treatment <- as.integer(time > sequence)
      exposure_period <- ifelse(treatment == 1L, time - sequence, 0L)
      delayed_tx <- numeric(Tp)
      treated <- exposure_period > 0L
      delayed_tx[treated] <- delay[exposure_period[treated]]

      rows[[row_id]] <- data.frame(
        sequence = sequence,
        cluster_within_sequence = cluster_within_sequence,
        cluster_id = cluster_id,
        time = time,
        treatment = treatment,
        exposure_period = exposure_period,
        delayed_tx = delayed_tx
      )
    }
  }

  data_base <- do.call(rbind, rows)
  data_base$cluster <- factor(data_base$cluster_id)
  data_base$period <- factor(data_base$time, levels = seq_len(Tp))
  data_base <- data_base[
    , c(
      "sequence", "cluster_within_sequence", "cluster", "cluster_id",
      "time", "period", "treatment", "exposure_period", "delayed_tx"
    )
  ]
  rownames(data_base) <- NULL

  balanced <- length(unique(clusters_per_sequence)) == 1L
  out <- list(
    data_base = data_base,
    I = sum(clusters_per_sequence),
    n_cells = nrow(data_base),
    Tp = Tp,
    clusters_per_sequence = clusters_per_sequence,
    no_clusters_per_step = if (balanced) clusters_per_sequence[1L] else NULL,
    balanced_clusters = balanced,
    delay = delay
  )
  class(out) <- "sw_ni_design"
  out
}
