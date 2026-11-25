#' Estimate power for a stepped-wedge non-inferiority design
#'
#' Main user-facing function for binary and continuous repeated
#' cross-sectional stepped-wedge designs. Version 0.2.0 adds batched PyTorch
#' GPU/CPU acceleration (ideal for Google Colab) alongside the reference `lme4`
#' engine, supporting fixed categorical period adjustment, user-specified
#' secular period effects, unequal sequence sizes, and unequal cluster-period
#' sample sizes.
#'
#' @param outcome Either `"binary"` or `"continuous"`.
#' @param seed Random-number seed.
#' @param trials Number of simulated trials.
#' @param Tp Number of time periods.
#' @param no_clusters_per_step One positive integer for a balanced design, or a
#'   vector of length `Tp - 1` for unequal numbers of clusters per sequence.
#' @param clusters_per_sequence Optional positive integer vector of length
#'   `Tp - 1`; takes precedence over `no_clusters_per_step`.
#' @param K Cluster-period sample size. May be a scalar, a vector of length `I`
#'   (one value per cluster), a vector of length `n_cells` (one value per
#'   cluster-period in design order), or an `I` by `Tp` matrix.
#' @param b1 Binary-outcome baseline log-odds intercept.
#' @param mu Continuous-outcome baseline mean.
#' @param sdw Continuous-outcome within-cluster individual-level standard
#'   deviation. Cluster-period means are generated with SD `sdw / sqrt(K_ij)`.
#' @param sdc Cluster-level random-intercept standard deviation.
#' @param sdcp Stationary AR(1) cluster-by-period random-effect marginal
#'   standard deviation (default `0`, i.e., shared cluster random intercept
#'   only). Positive values require a PyTorch backend (`"gpu"` or `"torch_cpu"`).
#' @param rho Autocorrelation parameter in `(-1, 1)` for the stationary AR(1)
#'   cluster-by-period random effect when `sdcp > 0`.
#' @param theta True treatment effect on the fitted-model scale.
#' @param margin Non-inferiority margin on the fitted-model scale. If `NULL`,
#'   defaults to 0.25 for binary outcomes and 0.15 for continuous outcomes.
#' @param delay Optional numeric vector of length `Tp - 1` defining treatment
#'   exposure by post-crossover period.
#' @param period_effects Optional secular effects on the outcome-model scale.
#'   Supply length `Tp`, or length `Tp - 1` with period 1 treated as the zero
#'   reference. Defaults to zero in all periods.
#' @param adjust_period Logical; include categorical period fixed effects in the
#'   fitted analysis model. The default is `TRUE`.
#' @param sig.level One-sided significance level.
#' @param REML Use REML for the continuous-outcome LMM.
#' @param failure_handling Handling of failed model fits: `"nonrejection"`
#'   counts failures as non-rejections (default), `"exclude"` omits them from
#'   the power denominator, and `"error"` stops if any fit fails.
#' @param parallel Logical; parallelize simulation replicates across CPU workers
#'   when using `backend = "lme4"`.
#' @param ncores Number of CPU workers when `parallel = TRUE` and
#'   `backend = "lme4"`.
#' @param return_details If `TRUE`, return power, Monte Carlo SE, fit-failure
#'   information, backend/device metadata, design totals, and normalized
#'   parameters.
#' @param cluster Optional existing cluster created by [parallel::makeCluster()].
#'   When supplied (with `backend = "auto"` or `"lme4"`), it takes precedence
#'   over `parallel` and `ncores`, and is not stopped by `sw_ni_power()`.
#' @param backend Simulation engine: `"auto"` (default; uses PyTorch GPU if
#'   available, else batched PyTorch CPU if available, else `"lme4"`), `"gpu"`
#'   (requires CUDA/MPS GPU via PyTorch), `"torch_cpu"` (batched PyTorch on
#'   CPU), or `"lme4"` (reference per-trial CPU fits via `lme4`).
#' @param batch_size Optional positive integer number of replicates processed
#'   simultaneously per batch on `"gpu"` / `"torch_cpu"` backends. Defaults to
#'   an automatic memory-safe batch size (`NULL`).
#' @param device Optional explicit PyTorch device string (for example `"cuda"`,
#'   `"cuda:0"`, `"mps"`, or `"cpu"`).
#'
#' @return Numeric power when `return_details = FALSE`; otherwise a list.
#' @export
sw_ni_power <- function(outcome = c("binary", "continuous"),
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
                        failure_handling = c("nonrejection", "exclude", "error"),
                        parallel = FALSE,
                        ncores = .default_ncores(),
                        return_details = FALSE,
                        cluster = NULL,
                        backend = c("auto", "gpu", "torch_cpu", "lme4"),
                        batch_size = NULL,
                        device = NULL,
                        sdcp = 0,
                        rho = 0) {
  outcome <- match.arg(outcome)
  failure_handling <- match.arg(failure_handling)
  backend <- match.arg(backend)
  if (is.null(margin)) margin <- .default_margin(outcome)

  .check_scalar_numeric(seed, "seed", integer = TRUE)
  .check_scalar_numeric(trials, "trials", lower = 1, integer = TRUE)
  .check_scalar_numeric(Tp, "Tp", lower = 2, integer = TRUE)
  .check_scalar_numeric(sdc, "sdc", lower = 0)
  .check_scalar_numeric(sdcp, "sdcp", lower = 0)
  .check_scalar_numeric(rho, "rho", lower = -0.9999, upper = 0.9999)
  .check_scalar_numeric(theta, "theta")
  .check_scalar_numeric(margin, "margin", lower = 0)
  .check_scalar_numeric(sig.level, "sig.level", lower = 0, upper = 1)
  .check_scalar_numeric(ncores, "ncores", lower = 1, integer = TRUE)
  .check_scalar_numeric(batch_size, "batch_size", lower = 1, integer = TRUE, allow_null = TRUE)
  if (!is.logical(parallel) || length(parallel) != 1L || is.na(parallel)) {
    stop("parallel must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(return_details) || length(return_details) != 1L || is.na(return_details)) {
    stop("return_details must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(cluster)) .validate_cluster(cluster)
  if (!is.logical(adjust_period) || length(adjust_period) != 1L || is.na(adjust_period)) {
    stop("adjust_period must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(REML) || length(REML) != 1L || is.na(REML)) {
    stop("REML must be TRUE or FALSE.", call. = FALSE)
  }
  if (isTRUE(adjust_period) && Tp < 3) {
    stop(
      "Tp must be at least 3 when adjust_period = TRUE because treatment and period are not separable with one crossover sequence.",
      call. = FALSE
    )
  }

  if (identical(outcome, "binary")) {
    .check_scalar_numeric(b1, "b1")
  } else {
    .check_scalar_numeric(mu, "mu")
    .check_scalar_numeric(sdw, "sdw", lower = 0)
  }

  design_obj <- sw_ni_design(
    Tp = Tp,
    no_clusters_per_step = no_clusters_per_step,
    clusters_per_sequence = clusters_per_sequence,
    delay = delay
  )
  K_vec <- .normalize_K(K, design_obj)
  period_effects <- .normalize_period_effects(period_effects, design_obj$Tp)
  trials <- as.integer(trials)

  resolved <- .resolve_execution_backend(
    backend = backend,
    device = device,
    cluster = cluster
  )

  if (identical(resolved$engine, "torch")) {
    payload <- .build_torch_scenario_payload(
      outcome = outcome,
      trials = trials,
      seed = seed,
      design_obj = design_obj,
      K_vec = K_vec,
      period_effects = period_effects,
      adjust_period = adjust_period,
      b1 = b1,
      mu = mu,
      sdw = sdw,
      sdc = sdc,
      theta = theta,
      margin = margin,
      sig.level = sig.level,
      REML = REML,
      batch_size = batch_size,
      device = resolved$device,
      sdcp = sdcp,
      rho = rho
    )
    torch_res <- .run_power_torch(payload, resolved$probe)
    summary <- .compute_power_summary_from_counts(
      n_rejections = torch_res$n_rejections,
      n_failed = torch_res$n_failed,
      trials = trials,
      failure_handling = failure_handling
    )
    backend_used <- resolved$mode
    device_used <- as.character(torch_res$device)
  } else {
    if (sdcp > 0) {
      stop(
        "AR(1) cluster-period random effects (sdcp > 0) require the PyTorch backend ('gpu' or 'torch_cpu').",
        call. = FALSE
      )
    }
    if (identical(outcome, "binary")) {
      control <- lme4::glmerControl(check.conv.singular = "ignore")
      sim_fun <- .sw_ni_simulate_binary_once
      sim_args <- list(
        design_obj = design_obj,
        K_vec = K_vec,
        period_effects = period_effects,
        adjust_period = adjust_period,
        b1 = b1,
        sdc = sdc,
        theta = theta,
        margin = margin,
        sig.level = sig.level,
        control = control
      )
    } else {
      control <- lme4::lmerControl(check.conv.singular = "ignore")
      sim_fun <- .sw_ni_simulate_continuous_once
      sim_args <- list(
        design_obj = design_obj,
        K_vec = K_vec,
        period_effects = period_effects,
        adjust_period = adjust_period,
        mu = mu,
        sdw = sdw,
        sdc = sdc,
        theta = theta,
        margin = margin,
        sig.level = sig.level,
        REML = REML,
        control = control
      )
    }

    one_run <- function(i, simulation_function, arguments) {
      do.call(simulation_function, arguments)
    }

    workers <- if (is.null(cluster) && isTRUE(parallel) && ncores > 1L) {
      .safe_ncores(ncores, tasks = trials)
    } else if (is.null(cluster)) {
      1L
    } else {
      length(cluster)
    }

    if (is.null(cluster) && workers <= 1L) {
      set.seed(seed)
      results <- vapply(
        seq_len(trials),
        one_run,
        integer(1L),
        simulation_function = sim_fun,
        arguments = sim_args
      )
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
      parallel::clusterSetRNGStream(cl, iseed = seed)
      results <- unlist(
        parallel::parLapplyLB(
          cl,
          seq_len(trials),
          one_run,
          simulation_function = sim_fun,
          arguments = sim_args
        ),
        use.names = FALSE
      )
    }

    summary <- .compute_power_summary(
      results = results,
      trials = trials,
      failure_handling = failure_handling
    )
    backend_used <- "lme4"
    device_used <- "cpu"
  }

  if (!return_details) return(summary$power)

  c(
    list(
      outcome = outcome,
      backend = backend_used,
      device = device_used
    ),
    summary,
    list(
      design = list(
        Tp = design_obj$Tp,
        clusters_per_sequence = design_obj$clusters_per_sequence,
        total_clusters = design_obj$I,
        cluster_period_cells = design_obj$n_cells,
        total_observations = sum(K_vec),
        balanced_clusters = design_obj$balanced_clusters
      ),
      parameters = list(
        K = K,
        normalized_K = K_vec,
        b1 = if (identical(outcome, "binary")) b1 else NULL,
        mu = if (identical(outcome, "continuous")) mu else NULL,
        sdw = if (identical(outcome, "continuous")) sdw else NULL,
        sdc = sdc,
        theta = theta,
        margin = margin,
        delay = design_obj$delay,
        period_effects = period_effects,
        adjust_period = adjust_period,
        sig.level = sig.level,
        REML = if (identical(outcome, "continuous")) REML else NULL
      )
    )
  )
}
