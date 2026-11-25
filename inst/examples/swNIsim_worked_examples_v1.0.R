# Worked power examples for the manuscript's archived swNIsim 1.0 engine.
# Run in a fresh R session with version 1.0 installed and on .libPaths().
# From a terminal in the directory that contains this script:
#   Rscript swNIsim_worked_examples_v1.0.R         # 100,000 per candidate
#   Rscript swNIsim_worked_examples_v1.0.R 10000   # pilot run
# In RStudio, source this file for the default run.

if (!requireNamespace("swNIsim", quietly = TRUE)) {
  stop("Install archived swNIsim 1.0 and its lme4 dependency first.", call. = FALSE)
}
version <- as.character(utils::packageVersion("swNIsim"))
if (!identical(version, "1.0")) {
  stop(sprintf("This script requires swNIsim 1.0; found %s.", version),
       call. = FALSE)
}

args <- if (interactive()) character(0L) else commandArgs(trailingOnly = TRUE)
B <- if (length(args)) suppressWarnings(as.integer(args[1L])) else 100000L
if (length(args) > 1L || is.na(B) || B < 1L ||
    (length(args) == 1L && !grepl("^[0-9]+$", args[1L]))) {
  stop("Usage: Rscript swNIsim_worked_examples_v1.0.R [replicates]", call. = FALSE)
}
RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")

run_example <- function(label, K, target, seeds, model) {
  stopifnot(length(K) == length(seeds))
  do.call(rbind, lapply(seq_along(K), function(i) {
    message(label, ": K = ", K[i], ", replicates = ", B)
    fit <- do.call(swNIsim::sw_ni_power, c(
      model,
      list(
        K = K[i], seed = seeds[i], trials = B,
        theta = 0, delay = NULL,
        period_effects = rep(0, model$Tp), adjust_period = TRUE,
        sig.level = 0.025, failure_handling = "nonrejection",
        parallel = FALSE, return_details = TRUE
      )
    ))
    data.frame(
      example = label, K = K[i], N = fit$design$total_observations,
      target = target, power = fit$power, mcse = fit$mcse,
      replicates = fit$n_trials, rejections = fit$n_rejections,
      failed = fit$n_failed, seed = seeds[i],
      near_target = abs(fit$power - target) <= 2 * fit$mcse
    )
  }))
}

therappp <- run_example(
  "TheraPPP overall", 4:6, 0.90, 20261110:20261112,
  list(outcome = "continuous", Tp = 9L, no_clusters_per_step = 2L,
       mu = 21, sdw = 10, sdc = 4.2, margin = 4.2, REML = TRUE)
)

ards <- run_example(
  "TheraPPP ARDS", 13:14, 0.90, 20261213:20261214,
  list(outcome = "continuous", Tp = 9L, no_clusters_per_step = 2L,
       mu = 11, sdw = 10, sdc = 1, margin = 2.2, REML = TRUE)
)

# PRIM-ER-based planning illustration: non-admission is favorable.
# Probabilities are conditional on a cluster random intercept of zero.
p_ref <- 0.30
p_min <- 0.24
icc <- 0.10
primer <- run_example(
  "PRIM-ER", 5:7, 0.85, 20261119:20261121,
  list(outcome = "binary", Tp = 31L, no_clusters_per_step = 1L,
       b1 = qlogis(p_ref), sdc = sqrt((icc / (1 - icc)) * pi^2 / 3),
       margin = qlogis(p_ref) - qlogis(p_min))
)

results <- rbind(therappp, ards, primer)
print(results, row.names = FALSE)
output <- sprintf("swNIsim_worked_examples_v1.0_%d_reps.csv", B)
utils::write.csv(results, output, row.names = FALSE)
message("Saved ", output)

message("Smallest tested K meeting each target (NA if none):")
print(vapply(split(results, results$example), function(x) {
  eligible <- x$K[which(x$power >= x$target)]
  if (length(eligible)) min(eligible) else NA_integer_
}, integer(1)))
if (any(results$near_target, na.rm = TRUE)) {
  message("At least one estimate is within two MCSEs of its target.",
          " Increase the replicate count if it affects the choice of K.")
}
