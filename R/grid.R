#' Build a balanced parameter grid
#'
#' Creates a conventional scalar parameter grid for [sw_ni_grid()]. Advanced
#' unequal sequence-size, unequal `K`, delay, or period-effect specifications can
#' be supplied to [sw_ni_grid()] through list-columns or as scalar defaults.
#'
#' @param outcome Either `"binary"` or `"continuous"`.
#' @param Tp,no_clusters_per_step,K,sdc,theta,margin,sig.level Parameter values
#'   crossed into a grid.
#' @param sdcp Cluster-by-period random-effect SD values; see [sw_ni_power()].
#' @param b1 Binary baseline log-odds values.
#' @param mu,sdw Continuous-outcome parameter values.
#' @param adjust_period Logical values indicating categorical period adjustment.
#' @param REML Continuous-outcome REML values.
#' @param nAGQ Binary-outcome values of `nAGQ` passed to [lme4::glmer()]; see
#'   [sw_ni_power()].
#' @param failure_handling Character values: `"nonrejection"`, `"exclude"`, or
#'   `"error"`.
#'
#' @return A data frame with one row per scenario.
#' @export
build_sw_ni_grid <- function(outcome = c("binary", "continuous"),
                             Tp = 5,
                             no_clusters_per_step = 4,
                             K = 20,
                             sdc = 1,
                             sdcp = 0,
                             theta = 0,
                             margin = NULL,
                             sig.level = 0.025,
                             b1 = 0.5,
                             mu = 5,
                             sdw = 2,
                             adjust_period = TRUE,
                             REML = TRUE,
                             nAGQ = 0,
                             failure_handling = "nonrejection") {
  outcome <- match.arg(outcome)
  if (is.null(margin)) margin <- .default_margin(outcome)

  common <- list(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    K = K,
    sdc = sdc,
    sdcp = sdcp,
    theta = theta,
    margin = margin,
    sig.level = sig.level,
    adjust_period = adjust_period,
    failure_handling = failure_handling
  )

  if (identical(outcome, "binary")) {
    common$b1 <- b1
    common$nAGQ <- nAGQ
  } else {
    common$mu <- mu
    common$sdw <- sdw
    common$REML <- REML
  }

  out <- do.call(
    expand.grid,
    c(common, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  )
  out$outcome <- outcome
  out
}

#' Run a parameter grid of stepped-wedge simulations
#'
#' Runs [sw_ni_power()] over every row of `params`. Columns take precedence over
#' scalar arguments. Advanced parameters such as `clusters_per_sequence`, `K`,
#' `delay`, and `period_effects` may be stored as list-columns.
#'
#' @param params A data frame, typically from [build_sw_ni_grid()]. Mixed outcome
#'   grids are not supported.
#' @param outcome Either `"binary"` or `"continuous"` unless `params$outcome`
#'   is present.
#' @param seed Base seed; row `i` uses `seed + i - 1`.
#' @param trials Number of simulated trials per scenario.
#' @param Tp,no_clusters_per_step,clusters_per_sequence,K,b1,mu,sdw,sdc,sdcp,theta,margin,delay,period_effects,adjust_period,fit_cp,sig.level,REML,nAGQ,failure_handling
#'   Defaults used when corresponding columns are absent. If any scenario has
#'   `adjust_period = FALSE`, one warning is given for the whole grid.
#' @param parallel Logical; parallelize across rows of `params`.
#' @param ncores Number of workers.
#' @param return_details If `TRUE`, append Monte Carlo SE, failure information,
#'   and design totals.
#'
#' @return A data frame containing the input parameters plus estimated power.
#' @export
sw_ni_grid <- function(params,
                       outcome = c("binary", "continuous"),
                       seed = 2026,
                       trials = 5000,
                       Tp = 5,
                       no_clusters_per_step = 4,
                       clusters_per_sequence = NULL,
                       K = 20,
                       b1 = 0.5,
                       mu = 5,
                       sdw = 2,
                       sdc = 1,
                       sdcp = 0,
                       theta = 0,
                       margin = NULL,
                       delay = NULL,
                       period_effects = NULL,
                       adjust_period = TRUE,
                       fit_cp = NULL,
                       sig.level = 0.025,
                       REML = TRUE,
                       nAGQ = 0,
                       failure_handling = "nonrejection",
                       parallel = TRUE,
                       ncores = .default_ncores(),
                       return_details = TRUE) {
  if (!is.data.frame(params)) {
    stop("params must be a data frame.", call. = FALSE)
  }
  if (nrow(params) < 1L) {
    stop("params must contain at least one row.", call. = FALSE)
  }

  if ("outcome" %in% names(params)) {
    outcomes_in_params <- unique(as.character(params$outcome))
    if (length(outcomes_in_params) != 1L) {
      stop("Mixed outcome grids are not supported. Run binary and continuous grids separately.", call. = FALSE)
    }
    outcome <- match.arg(outcomes_in_params, c("binary", "continuous"))
  } else {
    outcome <- match.arg(outcome)
    params$outcome <- outcome
  }
  if (is.null(margin)) margin <- .default_margin(outcome)

  unadjusted <- !vapply(
    seq_len(nrow(params)),
    function(i) isTRUE(.param_value(params, "adjust_period", i, adjust_period)),
    logical(1L)
  )
  if (any(unadjusted)) .warn_unadjusted()

  scenario_fun <- function(i) {
    out <- .muffle_unadjusted(sw_ni_power(
      outcome = as.character(params$outcome[i]),
      seed = seed + i - 1L,
      trials = trials,
      Tp = .param_value(params, "Tp", i, Tp),
      no_clusters_per_step = .param_value(params, "no_clusters_per_step", i, no_clusters_per_step),
      clusters_per_sequence = .param_value(params, "clusters_per_sequence", i, clusters_per_sequence),
      K = .param_value(params, "K", i, K),
      b1 = .param_value(params, "b1", i, b1),
      mu = .param_value(params, "mu", i, mu),
      sdw = .param_value(params, "sdw", i, sdw),
      sdc = .param_value(params, "sdc", i, sdc),
      sdcp = .param_value(params, "sdcp", i, sdcp),
      theta = .param_value(params, "theta", i, theta),
      margin = .param_value(params, "margin", i, margin),
      delay = .param_value(params, "delay", i, delay),
      period_effects = .param_value(params, "period_effects", i, period_effects),
      adjust_period = .param_value(params, "adjust_period", i, adjust_period),
      fit_cp = .param_value(params, "fit_cp", i, fit_cp),
      sig.level = .param_value(params, "sig.level", i, sig.level),
      REML = .param_value(params, "REML", i, REML),
      nAGQ = .param_value(params, "nAGQ", i, nAGQ),
      failure_handling = .param_value(params, "failure_handling", i, failure_handling),
      parallel = FALSE,
      return_details = TRUE
    ))

    c(
      power = out$power,
      mcse = out$mcse,
      n_failed = out$n_failed,
      failure_rate = out$failure_rate,
      n_successful_fits = out$n_successful_fits,
      effective_trials = out$effective_trials,
      total_clusters = out$design$total_clusters,
      cluster_period_cells = out$design$cluster_period_cells,
      total_observations = out$design$total_observations
    )
  }

  if (!parallel || ncores <= 1L) {
    res <- lapply(seq_len(nrow(params)), scenario_fun)
  } else {
    ncores <- min(as.integer(ncores), nrow(params))
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    res <- parallel::parLapplyLB(cl, seq_len(nrow(params)), scenario_fun)
  }

  res <- do.call(rbind, res)
  out <- params
  out$power <- as.numeric(res[, "power"])

  if (return_details) {
    out$mcse <- as.numeric(res[, "mcse"])
    out$n_failed <- as.integer(res[, "n_failed"])
    out$failure_rate <- as.numeric(res[, "failure_rate"])
    out$n_successful_fits <- as.integer(res[, "n_successful_fits"])
    out$effective_trials <- as.integer(res[, "effective_trials"])
    out$total_clusters <- as.integer(res[, "total_clusters"])
    out$cluster_period_cells <- as.integer(res[, "cluster_period_cells"])
    out$total_observations <- as.integer(res[, "total_observations"])
  }
  out
}
