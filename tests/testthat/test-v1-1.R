# Tests for changes introduced in swNIsim 1.1

test_that("build_margin requires an explicit method", {
  expect_error(build_margin("binary", p1 = 0.30, f1 = 0.80), "method must be specified")
  expect_error(build_margin("binary", method = "guess", p1 = 0.30), "method must be one of")
})

test_that("build_margin reproduces the manuscript's binary conversions", {
  m1 <- build_margin("binary", method = "preserve_control", p1 = 0.30, f1 = 0.80)
  expect_equal(m1$margin, qlogis(0.30) - qlogis(0.24))
  expect_equal(m1$b1, qlogis(0.30))
  expect_equal(m1$minimum_acceptable, 0.24)
  expect_equal(round(m1$margin, 3), 0.305)

  m2 <- build_margin("binary", method = "threshold", p1 = 0.30, p_min = 0.24)
  expect_equal(m2$margin, m1$margin)

  m3 <- build_margin("binary", method = "preserve_benefit", p1 = 0.60, p0 = 0.40, f2 = 0.50)
  expect_equal(m3$minimum_acceptable, 0.50)
  expect_equal(m3$margin, qlogis(0.60) - qlogis(0.50))

  m4 <- build_margin("binary", method = "direct", margin = 0.18, p1 = 0.622)
  expect_equal(m4$margin, 0.18)
  expect_equal(m4$b1, qlogis(0.622))
  expect_null(build_margin("binary", method = "direct", margin = 0.18)$b1)
})

test_that("build_margin reproduces the manuscript's continuous conversions", {
  expect_equal(build_margin("continuous", method = "preserve_control", mu = 21, f1 = 0.8)$margin, 4.2)
  expect_equal(build_margin("continuous", method = "preserve_benefit", mu = 21, mu0 = 15, f2 = 0.5)$margin, 3)
  expect_equal(build_margin("continuous", method = "threshold", mu = 11, mu_min = 8.8)$margin, 2.2)
  expect_equal(build_margin("continuous", method = "direct", margin = 0.15)$margin, 0.15)
  expect_null(build_margin("continuous", method = "direct", margin = 0.15)$b1)
})

test_that("build_margin has no silent defaults and rejects mixed or missing inputs", {
  expect_error(build_margin("binary", method = "preserve_control", p1 = 0.30), "requires: f1")
  expect_error(build_margin("binary", method = "preserve_benefit", p1 = 0.30, f2 = 0.5), "requires: p0")
  expect_error(
    build_margin("binary", method = "preserve_control", p1 = 0.30, f1 = 0.8, f2 = 0.5),
    "not used"
  )
  expect_error(build_margin("binary", method = "preserve_control", mu = 5, f1 = 0.8), "not used")
  expect_error(build_margin("continuous", method = "threshold", mu = 5, p_min = 0.2), "not used")
})

test_that("build_margin validates input ranges", {
  expect_error(build_margin("binary", method = "preserve_control", p1 = 1.2, f1 = 0.8), "p1 must be strictly")
  expect_error(build_margin("binary", method = "preserve_control", p1 = 0.3, f1 = 1), "f1 must be strictly")
  expect_error(build_margin("binary", method = "preserve_benefit", p1 = 0.3, p0 = 0.4, f2 = 0.5), "p0 must be")
  expect_error(build_margin("binary", method = "threshold", p1 = 0.3, p_min = 0.35), "p_min must be")
  expect_error(build_margin("continuous", method = "preserve_control", mu = -2, f1 = 0.8), "true zero")
  expect_error(build_margin("continuous", method = "preserve_benefit", mu = 5, mu0 = 6, f2 = 0.5), "mu0 must be")
  expect_error(build_margin("continuous", method = "threshold", mu = 5, mu_min = 5), "mu_min must be")
  expect_error(build_margin("continuous", method = "direct", margin = -0.1), "positive")
  expect_error(build_margin("binary", method = "preserve_control", p1 = c(0.3, 0.4), f1 = 0.8), "scalar")
})

test_that("adjust_period = FALSE warns, and the default does not", {
  skip_if_not_installed("lme4")
  expect_warning(
    sw_ni_power(outcome = "continuous", trials = 2, Tp = 3, no_clusters_per_step = 2,
                K = 5, adjust_period = FALSE, parallel = FALSE),
    class = "swNIsim_unadjusted_warning"
  )
  expect_no_warning(
    sw_ni_power(outcome = "continuous", trials = 2, Tp = 3, no_clusters_per_step = 2,
                K = 5, parallel = FALSE)
  )
})

test_that("sw_ni_grid gives one unadjusted warning for the whole grid", {
  skip_if_not_installed("lme4")
  g <- build_sw_ni_grid(outcome = "continuous", Tp = 3, no_clusters_per_step = 2,
                        K = c(5, 6), adjust_period = FALSE)
  warnings_seen <- 0L
  withCallingHandlers(
    sw_ni_grid(g, trials = 2, parallel = FALSE),
    swNIsim_unadjusted_warning = function(w) {
      warnings_seen <<- warnings_seen + 1L
      invokeRestart("muffleWarning")
    }
  )
  expect_equal(warnings_seen, 1L)
})

test_that("nAGQ is validated, passed to glmer, and reported", {
  skip_if_not_installed("lme4")
  expect_error(
    sw_ni_power(outcome = "binary", trials = 1, Tp = 3, nAGQ = -1, parallel = FALSE),
    "nAGQ"
  )
  expect_error(
    sw_ni_power(outcome = "binary", trials = 1, Tp = 3, nAGQ = 0.5, parallel = FALSE),
    "nAGQ"
  )
  x <- sw_ni_power(outcome = "binary", trials = 3, Tp = 3, no_clusters_per_step = 3, K = 20,
                   nAGQ = 1, parallel = FALSE, return_details = TRUE)
  expect_equal(x$parameters$nAGQ, 1L)
  expect_equal(x$n_failed, 0L)
  y <- sw_ni_power(outcome = "continuous", trials = 2, Tp = 3, no_clusters_per_step = 2,
                   K = 5, parallel = FALSE, return_details = TRUE)
  expect_null(y$parameters$nAGQ)
  g <- build_sw_ni_grid(outcome = "binary", Tp = 3, nAGQ = c(0, 1))
  expect_equal(sort(unique(g$nAGQ)), c(0, 1))
})

test_that("nAGQ = 0 (the default) gives exactly the same results as an explicit 0", {
  skip_if_not_installed("lme4")
  a <- sw_ni_power(outcome = "binary", seed = 5, trials = 20, Tp = 4, no_clusters_per_step = 2,
                   K = 15, parallel = FALSE)
  b <- sw_ni_power(outcome = "binary", seed = 5, trials = 20, Tp = 4, no_clusters_per_step = 2,
                   K = 15, nAGQ = 0, parallel = FALSE)
  expect_identical(a, b)
})

test_that("bisection finds the smallest value meeting the target", {
  f <- function(v) c(0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.98, 0.99)[v]
  expect_equal(swNIsim:::.bisect_smallest(f, 1L, 10L, 0.90), 6L)
  expect_equal(swNIsim:::.bisect_smallest(f, 1L, 10L, 0.05), 1L)
  expect_true(is.na(swNIsim:::.bisect_smallest(f, 1L, 10L, 0.999)))
  calls <- 0L
  g <- function(v) { calls <<- calls + 1L; f(v) }
  swNIsim:::.bisect_smallest(g, 1L, 10L, 0.90)
  expect_lte(calls, 6L)
})

test_that("sw_ni_sample_size rejects conflicting arguments", {
  expect_error(sw_ni_sample_size(search = "K", K = 10, outcome = "continuous"), "Do not supply K")
  expect_error(sw_ni_sample_size(search = "clusters", no_clusters_per_step = 2, outcome = "continuous"),
               "Do not supply")
  expect_error(sw_ni_sample_size(search = "K", trials = 10, seed = 1, return_details = TRUE,
                                 outcome = "continuous"), "return_details")
  expect_error(sw_ni_sample_size(target_power = 1.2, outcome = "continuous"), "target_power")
  expect_error(sw_ni_sample_size(search = "clusters", K = c(10, 20), outcome = "continuous"),
               "K must be a single number")
  expect_error(sw_ni_sample_size(lower = 10, upper = 5, outcome = "continuous"), "upper must be")
})

test_that("sw_ni_sample_size searches K and clusters and returns a consistent result", {
  skip_if_not_installed("lme4")
  res <- sw_ni_sample_size(
    target_power = 0.5, search = "K", lower = 2, upper = 40, trials = 40, seed = 3,
    outcome = "continuous", Tp = 4, no_clusters_per_step = 2,
    mu = 5, sdw = 2, sdc = 0.5, theta = 0, margin = 0.6, parallel = FALSE
  )
  expect_s3_class(res, "sw_ni_sample_size")
  expect_true(res$reached)
  ev <- res$evaluated
  expect_true(ev$power[ev$value == res$value] >= 0.5)
  if (res$value > 2) expect_true(ev$power[ev$value == res$value - 1] < 0.5)
  expect_output(print(res), "Smallest K")

  full <- sw_ni_sample_size(
    target_power = 0.5, search = "clusters", values = c(1, 2, 3), trials = 20, seed = 3,
    outcome = "continuous", Tp = 3, K = 10, mu = 5, sdw = 2, sdc = 0.5, theta = 0,
    margin = 0.6, parallel = FALSE
  )
  expect_equal(full$evaluated$value, 1:3)
  expect_equal(full$evaluated$total_clusters, c(2L, 4L, 6L))

  none <- sw_ni_sample_size(
    target_power = 0.999, search = "K", lower = 2, upper = 3, trials = 10, seed = 3,
    outcome = "continuous", Tp = 3, no_clusters_per_step = 1, margin = 0.01, parallel = FALSE
  )
  expect_false(none$reached)
  expect_output(print(none), "No K")
})
