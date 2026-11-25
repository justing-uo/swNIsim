#' Build a balanced parameter grid
#'
#' Creates a conventional scalar parameter grid for [sw_ni_grid()]. Advanced
#' unequal sequence-size, unequal `K`, delay, or period-effect specifications can
#' be supplied to [sw_ni_grid()] through list-columns or as scalar defaults.
#'
#' @param outcome Either `"binary"` or `"continuous"`.
#' @param Tp,no_clusters_per_step,K,sdc,theta,margin,sig.level Parameter values
#'   crossed into a grid.
#' @param b1 Binary baseline log-odds values.
#' @param mu,sdw Continuous-outcome parameter values.
#' @param adjust_period Logical values indicating categorical period adjustment.
#' @param REML Continuous-outcome REML values.
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
                             theta = 0,
                             margin = NULL,
                             sig.level = 0.025,
                             b1 = 0.5,
                             mu = 5,
                             sdw = 2,
                             adjust_period = TRUE,
                             REML = TRUE,
                             failure_handling = "nonrejection") {
  outcome <- match.arg(outcome)
  if (is.null(margin)) margin <- .default_margin(outcome)

  common <- list(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    K = K,
    sdc = sdc,
    theta = theta,
    margin = margin,
    sig.level = sig.level,
    adjust_period = adjust_period,
    failure_handling = failure_handling
  )

  if (identical(outcome, "binary")) {
    common$b1 <- b1
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
#' `delay`, and `period_effects` may be stored as list-columns. When PyTorch GPU
#' or batched CPU acceleration is active, scenarios execute in a shared PyTorch
#' session while batching replicates within each scenario on the accelerator.
#'
#' @param params A data frame, typically from [build_sw_ni_grid()]. Mixed outcome
#'   grids are not supported.
#' @param outcome Either `"binary"` or `"continuous"` unless `params$outcome`
#'   is present.
#' @param seed Base seed; row `i` uses `seed + i - 1`.
#' @param trials Number of simulated trials per scenario.
#' @param Tp,no_clusters_per_step,clusters_per_sequence,K,b1,mu,sdw,sdc,theta,
#' margin,delay,period_effects,adjust_period,sig.level,REML,failure_handling
#'   Defaults used when corresponding columns are absent.
#' @param parallel Logical; parallelize across rows of `params` when using
#'   `backend = "lme4"`.
#' @param ncores Number of CPU workers when `backend = "lme4"`.
#' @param return_details If `TRUE`, append Monte Carlo SE, failure information,
#'   and design totals.
#' @param cluster Optional existing cluster created by [parallel::makeCluster()].
#'   When supplied (with `backend = "auto"` or `"lme4"`), it takes precedence
#'   over `parallel` and `ncores`, and is not stopped by `sw_ni_grid()`.
#' @param backend Simulation engine: `"auto"`, `"gpu"`, `"torch_cpu"`, or
#'   `"lme4"`.
#' @param batch_size Optional positive integer number of replicates processed
#'   simultaneously per batch on `"gpu"` / `"torch_cpu"` backends.
#' @param device Optional explicit PyTorch device string.
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
                       theta = 0,
                       margin = NULL,
                       delay = NULL,
                       period_effects = NULL,
                       adjust_period = TRUE,
                       sig.level = 0.025,
                       REML = TRUE,
                       failure_handling = "nonrejection",
                       parallel = TRUE,
                       ncores = .default_ncores(),
                       return_details = TRUE,
                       cluster = NULL,
                       backend = c("auto", "gpu", "torch_cpu", "lme4"),
                       batch_size = NULL,
                       device = NULL) {
  if (!is.data.frame(params)) {
    stop("params must be a data frame.", call. = FALSE)
  }
  if (nrow(params) < 1L) {
    stop("params must contain at least one row.", call. = FALSE)
  }
  backend <- match.arg(backend)
  .check_scalar_numeric(seed, "seed", integer = TRUE)
  .check_scalar_numeric(trials, "trials", lower = 1, integer = TRUE)
  .check_scalar_numeric(ncores, "ncores", lower = 1, integer = TRUE)
  .check_scalar_numeric(batch_size, "batch_size", lower = 1, integer = TRUE, allow_null = TRUE)
  if (!is.logical(parallel) || length(parallel) != 1L || is.na(parallel)) {
    stop("parallel must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(return_details) || length(return_details) != 1L || is.na(return_details)) {
    stop("return_details must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(cluster)) .validate_cluster(cluster)

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

  resolved <- .resolve_execution_backend(
    backend = backend,
    device = device,
    cluster = cluster
  )

  if (identical(resolved$engine, "torch")) {
    n_scen <- nrow(params)
    trials_int <- as.integer(trials)
    payloads <- vector("list", n_scen)
    meta <- vector("list", n_scen)

    for (i in seq_len(n_scen)) {
      Tp_i <- .param_value(params, "Tp", i, Tp)
      cps_i <- .param_value(params, "clusters_per_sequence", i, clusters_per_sequence)
      ncps_i <- .param_value(params, "no_clusters_per_step", i, no_clusters_per_step)
      delay_i <- .param_value(params, "delay", i, delay)
      K_i <- .param_value(params, "K", i, K)
      pe_i <- .param_value(params, "period_effects", i, period_effects)
      adj_i <- .param_value(params, "adjust_period", i, adjust_period)
      b1_i <- .param_value(params, "b1", i, b1)
      mu_i <- .param_value(params, "mu", i, mu)
      sdw_i <- .param_value(params, "sdw", i, sdw)
      sdc_i <- .param_value(params, "sdc", i, sdc)
      th_i <- .param_value(params, "theta", i, theta)
      mar_i <- .param_value(params, "margin", i, margin)
      sig_i <- .param_value(params, "sig.level", i, sig.level)
      reml_i <- .param_value(params, "REML", i, REML)
      fh_i <- match.arg(
        as.character(.param_value(params, "failure_handling", i, failure_handling)),
        c("nonrejection", "exclude", "error")
      )

      .check_scalar_numeric(Tp_i, "Tp", lower = 2, integer = TRUE)
      .check_scalar_numeric(sdc_i, "sdc", lower = 0)
      .check_scalar_numeric(th_i, "theta")
      .check_scalar_numeric(mar_i, "margin", lower = 0)
      .check_scalar_numeric(sig_i, "sig.level", lower = 0, upper = 1)
      if (!is.logical(adj_i) || length(adj_i) != 1L || is.na(adj_i)) {
        stop("adjust_period must be TRUE or FALSE.", call. = FALSE)
      }
      if (!is.logical(reml_i) || length(reml_i) != 1L || is.na(reml_i)) {
        stop("REML must be TRUE or FALSE.", call. = FALSE)
      }
      if (isTRUE(adj_i) && Tp_i < 3) {
        stop(
          "Tp must be at least 3 when adjust_period = TRUE because treatment and period are not separable with one crossover sequence.",
          call. = FALSE
        )
      }
      if (identical(outcome, "binary")) {
        .check_scalar_numeric(b1_i, "b1")
      } else {
        .check_scalar_numeric(mu_i, "mu")
        .check_scalar_numeric(sdw_i, "sdw", lower = 0)
      }

      d_obj <- sw_ni_design(
        Tp = Tp_i,
        no_clusters_per_step = ncps_i,
        clusters_per_sequence = cps_i,
        delay = delay_i
      )
      K_vec <- .normalize_K(K_i, d_obj)
      pe_vec <- .normalize_period_effects(pe_i, d_obj$Tp)

      payloads[[i]] <- .build_torch_scenario_payload(
        outcome = outcome,
        trials = trials_int,
        seed = as.integer(seed) + i - 1L,
        design_obj = d_obj,
        K_vec = K_vec,
        period_effects = pe_vec,
        adjust_period = adj_i,
        b1 = b1_i,
        mu = mu_i,
        sdw = sdw_i,
        sdc = sdc_i,
        theta = th_i,
        margin = mar_i,
        sig.level = sig_i,
        REML = reml_i,
        batch_size = batch_size,
        device = resolved$device
      )
      meta[[i]] <- list(
        failure_handling = fh_i,
        total_clusters = d_obj$I,
        cluster_period_cells = d_obj$n_cells,
        total_observations = sum(K_vec)
      )
    }

    torch_list <- .run_grid_torch(payloads, resolved$probe)
    res <- lapply(seq_len(n_scen), function(i) {
      s <- .compute_power_summary_from_counts(
        n_rejections = torch_list[[i]]$n_rejections,
        n_failed = torch_list[[i]]$n_failed,
        trials = trials_int,
        failure_handling = meta[[i]]$failure_handling
      )
      c(
        power = s$power,
        mcse = s$mcse,
        n_failed = s$n_failed,
        failure_rate = s$failure_rate,
        n_successful_fits = s$n_successful_fits,
        effective_trials = s$effective_trials,
        total_clusters = meta[[i]]$total_clusters,
        cluster_period_cells = meta[[i]]$cluster_period_cells,
        total_observations = meta[[i]]$total_observations
      )
    })
    device_used <- if (length(torch_list)) as.character(torch_list[[1L]]$device) else resolved$device
    backend_used <- resolved$mode
  } else {
    scenario_fun <- function(i) {
      out <- sw_ni_power(
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
        theta = .param_value(params, "theta", i, theta),
        margin = .param_value(params, "margin", i, margin),
        delay = .param_value(params, "delay", i, delay),
        period_effects = .param_value(params, "period_effects", i, period_effects),
        adjust_period = .param_value(params, "adjust_period", i, adjust_period),
        sig.level = .param_value(params, "sig.level", i, sig.level),
        REML = .param_value(params, "REML", i, REML),
        failure_handling = .param_value(params, "failure_handling", i, failure_handling),
        parallel = FALSE,
        return_details = TRUE,
        backend = "lme4"
      )

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

    workers <- if (is.null(cluster) && isTRUE(parallel) && ncores > 1L) {
      .safe_ncores(ncores, tasks = nrow(params))
    } else if (is.null(cluster)) {
      1L
    } else {
      length(cluster)
    }

    if (is.null(cluster) && workers <= 1L) {
      res <- lapply(seq_len(nrow(params)), scenario_fun)
    } else {
      if (is.null(cluster)) {
        cl <- tryCatch(
          parallel::makePSOCKcluster(workers),
          error = function(e) {
            stop(
              "Could not start the PSOCK cluster with ", workers,
              " workers. Try a smaller ncores value. Original error: ",
              conditionMessage(e),
              call. = FALSE
            )
          }
        )
        on.exit(.stop_cluster(cl), add = TRUE)
      } else {
        cl <- cluster
      }
      res <- parallel::parLapplyLB(cl, seq_len(nrow(params)), scenario_fun)
    }
    backend_used <- "lme4"
    device_used <- "cpu"
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
  attr(out, "backend") <- backend_used
  attr(out, "device") <- device_used
  out
}
