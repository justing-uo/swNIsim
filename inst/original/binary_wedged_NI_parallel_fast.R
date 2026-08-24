# Fast stepped-wedge non-inferiority simulation for binary outcomes

library(lme4)


make_sw_design <- function(Tp = 5,
                           no_clusters_per_step = 4,
                           delay = NULL) {
  I <- (Tp - 1L) * no_clusters_per_step

  design <- expand.grid(
    time  = rep(seq_len(Tp), each = no_clusters_per_step),
    group = seq_len(Tp - 1L),
    KEEP.OUT.ATTRS = FALSE
  )

  design$treatment <- as.integer(design$time > design$group)
  design$cluster_id <- seq_len(no_clusters_per_step) +
    (design$group - 1L) * no_clusters_per_step

  if (is.null(delay)) {
    delay <- rep(1, Tp - 1L)
  }
  if (length(delay) != Tp - 1L) {
    stop("delay must be NULL or have length Tp - 1.")
  }

  # Original delayed-treatment coding preserved from uploaded version
  design$delayed_tx <- delay[design$time - design$treatment * design$group]
  design$delayed_tx[design$treatment == 0L] <- 0

  # Keep only model/simulation variables. Factors are created once.
  data_base <- data.frame(
    cluster    = factor(design$cluster_id),
    cluster_id = design$cluster_id,
    delayed_tx = design$delayed_tx
  )

  list(
    data_base = data_base,
    I = I,
    n_cells = nrow(data_base)
  )
}

# -----------------------------------------------------------------------------
# One simulated trial, using a precomputed design
# -----------------------------------------------------------------------------
wedged_inferiority_simulation_fast <- function(design_obj,
                                               K = 20,
                                               b1 = 0.5,
                                               sdc = 1,
                                               theta = 0,
                                               margin = 0.25,
                                               sig.level = 0.025,
                                               control = NULL) {
  if (is.null(control)) {
    control <- glmerControl(
      check.conv.singular = "ignore"
    )
  }

  data_sim <- design_obj$data_base

  alpha <- rnorm(design_obj$I, mean = 0, sd = sdc)
  eta <- b1 + alpha[data_sim$cluster_id] + theta * data_sim$delayed_tx
  p <- plogis(eta)

  data_sim$success <- rbinom(design_obj$n_cells, size = K, prob = p)
  data_sim$fail <- K - data_sim$success

  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      glmer(
        cbind(success, fail) ~ delayed_tx + (1 | cluster),
        data = data_sim,
        family = binomial(),
        nAGQ = 0,
        control = control
      )
    )),
    error = function(e) NULL
  )

  if (is.null(fit)) return(NA_integer_)

  fe <- fixef(fit)
  if (!"delayed_tx" %in% names(fe)) return(NA_integer_)

  V <- tryCatch(as.matrix(vcov(fit)), error = function(e) NULL)
  if (is.null(V) || !"delayed_tx" %in% rownames(V)) return(NA_integer_)

  coef_diff <- unname(fe["delayed_tx"])
  coef_se <- sqrt(V["delayed_tx", "delayed_tx"])

  if (!is.finite(coef_se) || coef_se <= 0) return(NA_integer_)

  z_NI <- (coef_diff + margin) / coef_se
  p_value <- 1 - pnorm(z_NI)

  as.integer(p_value < sig.level)
}

# -----------------------------------------------------------------------------
# Power for one scenario
#   parallel = TRUE is useful for a single scenario.
#   For a parameter grid, use run_parameter_grid_fast(), which parallelizes rows.
# -----------------------------------------------------------------------------
wedged_inferiority_power_fast <- function(seed = 2026,
                                          trials = 5000,
                                          Tp = 5,
                                          no_clusters_per_step = 4,
                                          K = 20,
                                          b1 = 0.5,
                                          sdc = 1,
                                          theta = 0,
                                          margin = 0.25,
                                          delay = NULL,
                                          sig.level = 0.025,
                                          parallel = FALSE,
                                          ncores = max(1L, parallel::detectCores() - 1L),
                                          return_details = FALSE) {
  design_obj <- make_sw_design(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    delay = delay
  )

  control <- glmerControl(
    check.conv.singular = "ignore"
  )

  one_run <- function(i) {
    wedged_inferiority_simulation_fast(
      design_obj = design_obj,
      K = K,
      b1 = b1,
      sdc = sdc,
      theta = theta,
      margin = margin,
      sig.level = sig.level,
      control = control
    )
  }

  if (!parallel || ncores <= 1L) {
    set.seed(seed)
    results <- vapply(seq_len(trials), one_run, integer(1L))
  } else {
    ncores <- min(as.integer(ncores), trials)
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)

    # Reproducible independent RNG streams across workers
    parallel::clusterSetRNGStream(cl, iseed = seed)
    parallel::clusterEvalQ(cl, library(lme4))
    parallel::clusterExport(
      cl,
      varlist = c("wedged_inferiority_simulation_fast"),
      envir = environment()
    )

    results <- unlist(
      parallel::parLapplyLB(cl, seq_len(trials), one_run),
      use.names = FALSE
    )
  }

  power <- mean(results, na.rm = TRUE)
  n_failed <- sum(is.na(results))

  if (return_details) {
    return(list(
      power = power,
      n_trials = trials,
      n_failed = n_failed,
      failure_rate = n_failed / trials
    ))
  }

  power
}

# Backward-compatible wrapper names, so old code still works
wedged_inferiority_simulation_binom <- function(Tp = 5,
                                                no_clusters_per_step = 4,
                                                K = 20,
                                                b1 = 0.5,
                                                sdc = 1,
                                                theta = 0,
                                                margin = 0.25,
                                                delay = NULL,
                                                sig.level = 0.025) {
  design_obj <- make_sw_design(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    delay = delay
  )

  wedged_inferiority_simulation_fast(
    design_obj = design_obj,
    K = K,
    b1 = b1,
    sdc = sdc,
    theta = theta,
    margin = margin,
    sig.level = sig.level
  )
}

wedged_inferiority_power <- function(seed = 2026,
                                     trials = 5000,
                                     Tp = 5,
                                     no_clusters_per_step = 4,
                                     K = 20,
                                     b1 = 0.5,
                                     sdc = 1,
                                     theta = 0,
                                     margin = 0.25,
                                     delay = NULL,
                                     sig.level = 0.025,
                                     parallel = FALSE,
                                     ncores = max(1L, parallel::detectCores() - 1L)) {
  wedged_inferiority_power_fast(
    seed = seed,
    trials = trials,
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    K = K,
    b1 = b1,
    sdc = sdc,
    theta = theta,
    margin = margin,
    delay = delay,
    sig.level = sig.level,
    parallel = parallel,
    ncores = ncores,
    return_details = FALSE
  )
}

# -----------------------------------------------------------------------------
# Parameter grid runner
# Parallelizes across scenarios instead of opening a cluster for every row.
# -----------------------------------------------------------------------------
run_parameter_grid_fast <- function(params,
                                    seed = 2026,
                                    trials = 5000,
                                    b1 = 0.5,
                                    theta = 0,
                                    margin = 0.25,
                                    delay = NULL,
                                    sig.level = 0.025,
                                    parallel = TRUE,
                                    ncores = max(1L, parallel::detectCores() - 1L),
                                    return_details = TRUE) {
  stopifnot(all(c("sdc", "Tp", "no_clusters_per_step", "K") %in% names(params)))

  scenario_fun <- function(i) {
    out <- wedged_inferiority_power_fast(
      seed = seed + i - 1L,
      trials = trials,
      Tp = params$Tp[i],
      no_clusters_per_step = params$no_clusters_per_step[i],
      K = params$K[i],
      b1 = b1,
      sdc = params$sdc[i],
      theta = theta,
      margin = margin,
      delay = delay,
      sig.level = sig.level,
      parallel = FALSE,
      return_details = TRUE
    )

    c(power = out$power, n_failed = out$n_failed, failure_rate = out$failure_rate)
  }

  if (!parallel || ncores <= 1L) {
    res <- lapply(seq_len(nrow(params)), scenario_fun)
  } else {
    ncores <- min(as.integer(ncores), nrow(params))
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)

    parallel::clusterEvalQ(cl, library(lme4))
    parallel::clusterExport(
      cl,
      varlist = c(
        "params", "seed", "trials", "b1", "theta", "margin", "delay",
        "sig.level", "wedged_inferiority_power_fast",
        "wedged_inferiority_simulation_fast", "make_sw_design"
      ),
      envir = environment()
    )

    res <- parallel::parLapplyLB(cl, seq_len(nrow(params)), scenario_fun)
  }

  res <- do.call(rbind, res)
  out <- params
  out$power <- as.numeric(res[, "power"])

  if (return_details) {
    out$n_failed <- as.integer(res[, "n_failed"])
    out$failure_rate <- as.numeric(res[, "failure_rate"])
  }

  out
}

# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------
# Set this to TRUE when you want the full grid to run. Keeping it FALSE makes it
# safe to source() this file without accidentally launching 36 x 5000 GLMM fits.
RUN_EXAMPLE <- FALSE

if (RUN_EXAMPLE) {
  params <- data.frame(
    sdc = rep(c(1, 1.5, 2), each = 12),
    Tp  = rep(rep(c(7, 6, 5, 4), each = 3), times = 3),
    no_clusters_per_step = rep(c(10, 8, 4, 12, 10, 8, 15, 12, 10, 20, 16, 10), times = 3),
    K = rep(c(20, 25, 50, 20, 24, 30, 20, 25, 30, 20, 25, 40), times = 3)
  )

  # Use a smaller number while testing, e.g., trials = 100.
  params_results <- run_parameter_grid_fast(
    params = params,
    seed = 2026,
    trials = 5000,
    parallel = TRUE,
    ncores = max(1L, parallel::detectCores() - 1L)
  )

  print(params_results)

  # Optional export to Word
  if (requireNamespace("officer", quietly = TRUE) &&
      requireNamespace("flextable", quietly = TRUE)) {
    library(officer)
    library(flextable)

    ft <- flextable(params_results)
    ft <- autofit(ft)

    doc <- read_docx()
    doc <- body_add_par(doc, "Wedged Inferiority Results", style = "heading 1")
    doc <- body_add_flextable(doc, ft)

    print(doc, target = "wedged_inferiority_results.docx")
  }
}
