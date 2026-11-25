test_that("sw_ni_gpu_status returns structured status and prints cleanly", {
  st <- sw_ni_gpu_status(refresh = TRUE)
  expect_s3_class(st, "sw_ni_gpu_status")
  expect_true(all(c(
    "torch_available", "gpu_available", "cuda_available",
    "bridge", "lme4_available", "default_backend"
  ) %in% names(st)))
  expect_output(print(st), "swNIsim Backend Status")
})

test_that("explicit gpu backend errors informatively when no GPU is present", {
  st <- sw_ni_gpu_status()
  if (isTRUE(st$gpu_available)) {
    skip("GPU is available on this machine; skipping no-GPU error check.")
  }
  expect_error(
    sw_ni_power(outcome = "continuous", trials = 2, backend = "gpu"),
    "GPU|torch"
  )
})

test_that("minimal JSON encoder/decoder roundtrips scenario payloads without jsonlite", {
  d <- sw_ni_design(Tp = 4, clusters_per_sequence = c(1, 2, 1))
  K_vec <- swNIsim:::.normalize_K(15L, d)
  pe_vec <- swNIsim:::.normalize_period_effects(c(0, 0.1, 0.2, 0.3), 4)
  payload <- swNIsim:::.build_torch_scenario_payload(
    outcome = "continuous",
    trials = 10L,
    seed = 2026L,
    design_obj = d,
    K_vec = K_vec,
    period_effects = pe_vec,
    adjust_period = TRUE,
    b1 = 0.5,
    mu = 5,
    sdw = 2,
    sdc = 1,
    theta = 0,
    margin = 0.15,
    sig.level = 0.025,
    REML = TRUE,
    batch_size = 5L,
    device = "cpu"
  )
  js <- swNIsim:::.to_json_minimal(payload)
  expect_true(is.character(js) && nzchar(js))
  expect_match(js, '"outcome":"continuous"')
  expect_match(js, '"K_vec":\\[')
})

test_that("PyTorch backend agrees with lme4 when torch is available", {
  st <- sw_ni_gpu_status()
  if (!isTRUE(st$torch_available)) {
    skip("Python torch is not installed in this test environment.")
  }
  skip_if_not_installed("lme4")

  p_torch <- sw_ni_power(
    outcome = "continuous",
    seed = 2026,
    trials = 400,
    Tp = 5,
    clusters_per_sequence = c(2, 3, 2, 2),
    K = 20,
    mu = 5,
    sdw = 2,
    sdc = 1,
    margin = 0.8,
    backend = "torch_cpu",
    return_details = TRUE
  )
  p_lme4 <- sw_ni_power(
    outcome = "continuous",
    seed = 3026,
    trials = 400,
    Tp = 5,
    clusters_per_sequence = c(2, 3, 2, 2),
    K = 20,
    mu = 5,
    sdw = 2,
    sdc = 1,
    margin = 0.8,
    backend = "lme4",
    return_details = TRUE
  )
  se_diff <- sqrt(p_torch$mcse^2 + p_lme4$mcse^2)
  expect_lt(abs(p_torch$power - p_lme4$power), 4 * max(se_diff, 0.02))
})
