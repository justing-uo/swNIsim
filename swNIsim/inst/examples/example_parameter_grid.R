library(swNIsim)

# Conventional balanced grid
cont_params <- build_sw_ni_grid(
  outcome = "continuous",
  Tp = c(5, 6),
  no_clusters_per_step = c(4, 8),
  K = c(20, 30),
  sdc = c(0.5, 1),
  mu = 5,
  sdw = 2,
  theta = 0,
  margin = 0.15,
  adjust_period = TRUE
)

cont_res <- sw_ni_grid(
  params = cont_params,
  seed = 2026,
  trials = 500,
  parallel = FALSE
)
print(cont_res)

# Advanced grid with unequal sequence sizes and period effects
advanced <- data.frame(
  outcome = c("continuous", "continuous"),
  sdc = c(1, 1.3),
  stringsAsFactors = FALSE
)
advanced$clusters_per_sequence <- I(list(c(3, 4, 5, 4), c(4, 4, 4, 4)))
advanced$period_effects <- I(list(c(0, .1, .1, .2, .2), rep(0, 5)))
advanced$K <- I(list(20L, 25L))

advanced_res <- sw_ni_grid(
  advanced,
  Tp = 5,
  mu = 5,
  sdw = 2,
  margin = 0.15,
  trials = 500,
  parallel = FALSE
)
print(advanced_res)
