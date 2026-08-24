test_that("balanced and unequal design objects are built", {
  d1 <- sw_ni_design(Tp = 5, no_clusters_per_step = 4)
  expect_s3_class(d1, "sw_ni_design")
  expect_equal(d1$I, 16)
  expect_equal(d1$n_cells, 80)
  expect_true(all(c("time", "period", "sequence", "cluster", "delayed_tx") %in% names(d1$data_base)))

  d2 <- sw_ni_design(Tp = 4, clusters_per_sequence = c(1, 2, 3))
  expect_equal(d2$I, 6)
  expect_equal(d2$n_cells, 24)
  expect_false(d2$balanced_clusters)
  expect_equal(d2$clusters_per_sequence, c(1L, 2L, 3L))
})

test_that("K normalization supports scalar, cluster vector, cell vector, and matrix", {
  d <- sw_ni_design(Tp = 4, clusters_per_sequence = c(1, 2, 3))
  expect_equal(swNIsim:::.normalize_K(20, d), rep(20L, d$n_cells))
  expect_equal(length(swNIsim:::.normalize_K(rep(20L, d$I), d)), d$n_cells)
  expect_equal(swNIsim:::.normalize_K(rep(21L, d$n_cells), d), rep(21L, d$n_cells))
  km <- matrix(22L, nrow = d$I, ncol = d$Tp)
  expect_equal(swNIsim:::.normalize_K(km, d), rep(22L, d$n_cells))
})

test_that("period effects accept Tp and Tp minus one forms", {
  expect_equal(swNIsim:::.normalize_period_effects(NULL, 5), rep(0, 5))
  expect_equal(swNIsim:::.normalize_period_effects(c(.1, .2, .3, .4), 5), c(0, .1, .2, .3, .4))
})

test_that("grid builder works", {
  g <- build_sw_ni_grid(outcome = "binary", Tp = c(5, 6), K = c(20, 30))
  expect_true(is.data.frame(g))
  expect_true(all(g$outcome == "binary"))
  expect_true(all(g$adjust_period))
})

test_that("single-scenario continuous interface returns version 0.1.2 details", {
  skip_if_not_installed("lme4")
  x <- sw_ni_power(
    outcome = "continuous",
    trials = 2,
    Tp = 4,
    clusters_per_sequence = c(2, 2, 2),
    K = 20,
    period_effects = c(0, .1, .1, .2),
    parallel = FALSE,
    return_details = TRUE
  )
  expect_true(is.list(x))
  expect_true(all(c("power", "mcse", "n_failed", "design") %in% names(x)))
  expect_equal(x$design$total_observations, 4 * 6 * 20)
})
