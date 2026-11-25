library(swNIsim)

if (requireNamespace("testthat", quietly = TRUE)) {
  testthat::test_check("swNIsim")
} else {
  d <- sw_ni_design(Tp = 4, clusters_per_sequence = c(1, 2, 3))
  stopifnot(inherits(d, "sw_ni_design"), d$I == 6L, d$n_cells == 24L)

  g <- build_sw_ni_grid(outcome = "continuous", Tp = c(4, 5), K = c(10, 20))
  stopifnot(is.data.frame(g), nrow(g) == 4L)

  st <- sw_ni_gpu_status()
  stopifnot(inherits(st, "sw_ni_gpu_status"))

  if (requireNamespace("lme4", quietly = TRUE) || isTRUE(st$torch_available)) {
    ans <- sw_ni_power(
      outcome = "continuous",
      trials = 2,
      Tp = 4,
      clusters_per_sequence = c(2, 2, 2),
      K = 20,
      parallel = FALSE,
      return_details = TRUE
    )
    stopifnot(is.list(ans), ans$design$total_observations == 480L)
  }
}
