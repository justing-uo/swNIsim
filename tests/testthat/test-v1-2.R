# Tests for the nested exchangeable option introduced in swNIsim 1.2

test_that("sdcp = 0 reproduces the exchangeable results exactly", {
  skip_if_not_installed("lme4")
  for (outcome in c("continuous", "binary")) {
    a <- sw_ni_power(outcome = outcome, seed = 7, trials = 15, Tp = 4, no_clusters_per_step = 2,
                     K = 12, parallel = FALSE)
    b <- sw_ni_power(outcome = outcome, seed = 7, trials = 15, Tp = 4, no_clusters_per_step = 2,
                     K = 12, sdcp = 0, parallel = FALSE)
    expect_identical(a, b)
  }
})

test_that("correlation summaries follow the nested exchangeable formulas", {
  cc <- swNIsim:::.correlation_summary("continuous", sdc = sqrt(0.5), sdcp = sqrt(0.5), sdw = 2)
  expect_equal(cc$within_period_icc, 0.2)
  expect_equal(cc$between_period_icc, 0.1)
  expect_equal(cc$cluster_autocorrelation, 0.5)
  expect_equal(cc$structure, "nested exchangeable")

  ce <- swNIsim:::.correlation_summary("continuous", sdc = 1, sdcp = 0, sdw = 2)
  expect_equal(ce$within_period_icc, ce$between_period_icc)
  expect_equal(ce$cluster_autocorrelation, 1)
  expect_equal(ce$structure, "exchangeable")

  cb <- swNIsim:::.correlation_summary("binary", sdc = 1, sdcp = 0, sdw = 2)
  expect_equal(cb$within_period_icc, 1 / (1 + pi^2 / 3))
  expect_equal(cb$scale, "latent logistic")
})

test_that("fit_cp resolves automatically and is validated", {
  expect_true(swNIsim:::.resolve_fit_cp(NULL, "binary", 0.5))
  expect_false(swNIsim:::.resolve_fit_cp(NULL, "binary", 0))
  expect_false(swNIsim:::.resolve_fit_cp(FALSE, "binary", 0.5))
  expect_true(swNIsim:::.resolve_fit_cp(NA, "binary", 0.5))
  expect_false(swNIsim:::.resolve_fit_cp(NULL, "continuous", 0.5))
  expect_error(swNIsim:::.resolve_fit_cp(TRUE, "continuous", 0.5), "binary outcomes only")
  expect_error(swNIsim:::.resolve_fit_cp("yes", "binary", 0.5), "fit_cp must be")
})

test_that("continuous outcomes run with sdcp > 0 and report the structure", {
  skip_if_not_installed("lme4")
  x <- sw_ni_power(outcome = "continuous", trials = 3, Tp = 4, no_clusters_per_step = 2, K = 10,
                   sdc = 0.7, sdcp = 0.7, parallel = FALSE, return_details = TRUE)
  expect_equal(x$n_failed, 0L)
  expect_equal(x$correlation$structure, "nested exchangeable")
  expect_equal(x$parameters$sdcp, 0.7)
  expect_null(x$parameters$fit_cp)
  expect_error(
    sw_ni_power(outcome = "continuous", trials = 1, Tp = 3, sdcp = 0.5, fit_cp = TRUE, parallel = FALSE),
    "binary outcomes only"
  )
  expect_error(sw_ni_power(outcome = "continuous", trials = 1, Tp = 3, sdcp = -1, parallel = FALSE), "sdcp")
})

test_that("unequal cluster-period sizes with sdcp > 0 warn for continuous outcomes", {
  skip_if_not_installed("lme4")
  d <- sw_ni_design(Tp = 3, no_clusters_per_step = 2)
  km <- matrix(c(10L, 20L, 10L), nrow = d$I, ncol = 3, byrow = TRUE)
  expect_warning(
    sw_ni_power(outcome = "continuous", trials = 2, Tp = 3, no_clusters_per_step = 2, K = km,
                sdcp = 0.5, parallel = FALSE),
    "approximate"
  )
  expect_no_warning(
    sw_ni_power(outcome = "continuous", trials = 2, Tp = 3, no_clusters_per_step = 2, K = km,
                parallel = FALSE)
  )
})

test_that("binary outcomes fit the cluster-by-period effect when asked", {
  skip_if_not_installed("lme4")
  x <- sw_ni_power(outcome = "binary", trials = 3, Tp = 4, no_clusters_per_step = 2, K = 25,
                   sdc = 0.5, sdcp = 0.5, parallel = FALSE, return_details = TRUE)
  expect_true(x$parameters$fit_cp)
  expect_equal(x$n_failed, 0L)
  y <- sw_ni_power(outcome = "binary", trials = 3, Tp = 4, no_clusters_per_step = 2, K = 25,
                   sdc = 0.5, sdcp = 0.5, fit_cp = FALSE, parallel = FALSE, return_details = TRUE)
  expect_false(y$parameters$fit_cp)
  z <- sw_ni_power(outcome = "binary", trials = 3, Tp = 4, no_clusters_per_step = 2, K = 25,
                   sdc = 0.5, sdcp = 0.5, nAGQ = 1, parallel = FALSE, return_details = TRUE)
  expect_equal(z$n_failed, 0L)
  expect_error(
    sw_ni_power(outcome = "binary", trials = 1, Tp = 3, sdcp = 0.5, nAGQ = 2, parallel = FALSE),
    "nAGQ > 1"
  )
})

test_that("grids and the sample-size search pass the new arguments through", {
  skip_if_not_installed("lme4")
  g <- build_sw_ni_grid(outcome = "continuous", Tp = 3, no_clusters_per_step = 2, K = 8,
                        sdcp = c(0, 0.5))
  expect_equal(sort(unique(g$sdcp)), c(0, 0.5))
  r <- sw_ni_grid(g, trials = 2, parallel = FALSE)
  expect_equal(nrow(r), 2L)

  gb <- build_sw_ni_grid(outcome = "binary", Tp = 3, no_clusters_per_step = 2, K = 20, sdcp = 0.5)
  gb$fit_cp <- c(FALSE)
  rb <- sw_ni_grid(gb, trials = 2, parallel = FALSE)
  expect_equal(nrow(rb), 1L)

  s <- sw_ni_sample_size(
    target_power = 0.5, search = "K", values = c(5, 10), trials = 5, seed = 1,
    outcome = "continuous", Tp = 3, no_clusters_per_step = 2, sdcp = 0.5, parallel = FALSE
  )
  expect_equal(s$evaluated$value, c(5L, 10L))
})
