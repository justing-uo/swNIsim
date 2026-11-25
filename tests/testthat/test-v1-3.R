test_that("rho is validated and reported", {
  expect_error(sw_ni_power(outcome = "continuous", trials = 2, rho = 1), "rho")
  cs <- swNIsim:::.correlation_summary("continuous", sdc = 1, sdcp = 1, sdw = 2, rho = 0.5)
  expect_identical(cs$structure, "AR(1) cluster-by-period")
  expect_equal(cs$between_period_icc, (1 + 0.5) / (1 + 1 + 4))
  cs0 <- swNIsim:::.correlation_summary("continuous", sdc = 1, sdcp = 1, sdw = 2)
  expect_identical(cs0$structure, "nested exchangeable")
})

test_that("auto chooses lme4 for settings only lme4 supports", {
  skip_if_not_installed("lme4")
  a <- sw_ni_power(outcome = "binary", trials = 3, Tp = 4, no_clusters_per_step = 2,
                   sdcp = 0.5, return_details = TRUE)
  expect_identical(a$backend, "lme4")
  b <- sw_ni_power(outcome = "binary", trials = 3, Tp = 4, no_clusters_per_step = 2,
                   nAGQ = 1, return_details = TRUE)
  expect_identical(b$backend, "lme4")
})

test_that("engine-specific settings give clear errors", {
  expect_error(
    sw_ni_power(outcome = "continuous", trials = 2, sdcp = 0.5, rho = 0.5, backend = "lme4"),
    "require backend"
  )
  expect_error(
    sw_ni_power(outcome = "binary", trials = 2, nAGQ = 1, backend = "torch_cpu"),
    "only available with backend = 'lme4'"
  )
  expect_error(
    sw_ni_power(outcome = "binary", trials = 2, sdcp = 0.5, rho = 0.5, fit_cp = TRUE),
    "requires the lme4 engine"
  )
})

test_that("backend = 'lme4' with sdcp = 0 matches the 1.2 engine draw for draw", {
  skip_if_not_installed("lme4")
  args <- list(outcome = "continuous", seed = 11, trials = 20, Tp = 4,
               no_clusters_per_step = 2, K = 10, return_details = TRUE)
  x <- do.call(sw_ni_power, c(args, backend = "lme4"))
  y <- do.call(sw_ni_power, c(args, backend = "lme4", sdcp = 0, rho = 0))
  expect_identical(x$n_rejections, y$n_rejections)
  expect_identical(x$backend, "lme4")
})
