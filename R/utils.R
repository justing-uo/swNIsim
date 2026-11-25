.default_ncores <- function() {
  n <- suppressWarnings(parallel::detectCores())
  if (length(n) != 1L || is.na(n) || n < 2L) 1L else as.integer(n - 1L)
}

.default_margin <- function(outcome) {
  outcome <- match.arg(outcome, c("binary", "continuous"))
  if (identical(outcome, "binary")) 0.25 else 0.15
}

.check_scalar_numeric <- function(x, name, lower = -Inf, upper = Inf,
                                  integer = FALSE, allow_null = FALSE) {
  if (is.null(x) && allow_null) return(invisible(TRUE))
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    stop(name, " must be a finite numeric scalar.", call. = FALSE)
  }
  if (integer && x != as.integer(x)) {
    stop(name, " must be an integer scalar.", call. = FALSE)
  }
  if (x < lower || x > upper) {
    stop(name, " must be between ", lower, " and ", upper, ".", call. = FALSE)
  }
  invisible(TRUE)
}

.check_positive_integer_vector <- function(x, name, required_length = NULL) {
  if (!is.numeric(x) || length(x) < 1L || any(!is.finite(x)) ||
      any(x < 1) || any(x != as.integer(x))) {
    stop(name, " must contain positive integers.", call. = FALSE)
  }
  if (!is.null(required_length) && length(x) != required_length) {
    stop(name, " must have length ", required_length, ".", call. = FALSE)
  }
  invisible(TRUE)
}

.param_value <- function(params, name, i, default) {
  if (!name %in% names(params)) return(default)
  x <- params[[name]]
  if (is.list(x)) x[[i]] else x[i]
}

.normalize_delay <- function(delay, Tp) {
  if (is.null(delay)) return(rep(1, Tp - 1L))
  if (!is.numeric(delay) || length(delay) != Tp - 1L || any(!is.finite(delay))) {
    stop("delay must be NULL or a finite numeric vector with length Tp - 1.", call. = FALSE)
  }
  as.numeric(delay)
}

.normalize_period_effects <- function(period_effects, Tp) {
  if (is.null(period_effects)) return(rep(0, Tp))
  if (!is.numeric(period_effects) || any(!is.finite(period_effects))) {
    stop("period_effects must be NULL or a finite numeric vector.", call. = FALSE)
  }
  if (length(period_effects) == Tp - 1L) {
    period_effects <- c(0, period_effects)
  }
  if (length(period_effects) != Tp) {
    stop("period_effects must have length Tp, or Tp - 1 with period 1 used as the zero reference.", call. = FALSE)
  }
  as.numeric(period_effects)
}

.resolve_clusters_per_sequence <- function(Tp,
                                           no_clusters_per_step,
                                           clusters_per_sequence = NULL) {
  if (!is.null(clusters_per_sequence)) {
    .check_positive_integer_vector(
      clusters_per_sequence,
      "clusters_per_sequence",
      required_length = Tp - 1L
    )
    return(as.integer(clusters_per_sequence))
  }

  .check_positive_integer_vector(no_clusters_per_step, "no_clusters_per_step")
  if (length(no_clusters_per_step) == 1L) {
    return(rep(as.integer(no_clusters_per_step), Tp - 1L))
  }
  if (length(no_clusters_per_step) != Tp - 1L) {
    stop(
      "no_clusters_per_step must be a scalar or a vector with length Tp - 1. ",
      "Alternatively, supply clusters_per_sequence.",
      call. = FALSE
    )
  }
  as.integer(no_clusters_per_step)
}

.normalize_K <- function(K, design_obj) {
  n_clusters <- design_obj$I
  n_cells <- design_obj$n_cells
  Tp <- design_obj$Tp
  dat <- design_obj$data_base

  if (is.matrix(K)) {
    if (!identical(dim(K), c(n_clusters, Tp))) {
      stop(
        "When K is a matrix, it must have I rows (clusters) and Tp columns (periods): ",
        n_clusters, " x ", Tp, ".",
        call. = FALSE
      )
    }
    K_vec <- K[cbind(dat$cluster_id, dat$time)]
  } else {
    if (!is.numeric(K) || length(K) < 1L || any(!is.finite(K))) {
      stop("K must be a finite numeric scalar, vector, or I-by-Tp matrix.", call. = FALSE)
    }
    if (length(K) == 1L) {
      K_vec <- rep(K, n_cells)
    } else if (length(K) == n_clusters) {
      K_vec <- K[dat$cluster_id]
    } else if (length(K) == n_cells) {
      K_vec <- K
    } else {
      stop(
        "A vector K must have length 1, I (one value per cluster), or n_cells ",
        "(one value per cluster-period).",
        call. = FALSE
      )
    }
  }

  .check_positive_integer_vector(K_vec, "K")
  as.integer(K_vec)
}

.parse_numeric_text <- function(x, name, allow_blank = TRUE) {
  if (is.null(x) || !nzchar(trimws(x))) {
    if (allow_blank) return(NULL)
    stop(name, " cannot be blank.", call. = FALSE)
  }
  pieces <- strsplit(x, "[,;[:space:]]+")[[1L]]
  pieces <- pieces[nzchar(pieces)]
  values <- suppressWarnings(as.numeric(pieces))
  if (length(values) < 1L || any(!is.finite(values))) {
    stop(name, " must be a comma- or space-separated numeric value or vector.", call. = FALSE)
  }
  values
}

.compute_power_summary <- function(results,
                                   trials,
                                   failure_handling = c("nonrejection", "exclude", "error")) {
  failure_handling <- match.arg(failure_handling)
  n_failed <- sum(is.na(results))
  n_successful <- trials - n_failed

  if (failure_handling == "error" && n_failed > 0L) {
    stop(n_failed, " model fit(s) failed. Set failure_handling to 'nonrejection' or 'exclude'.", call. = FALSE)
  }

  n_rejections <- sum(results == 1L, na.rm = TRUE)
  if (failure_handling == "exclude") {
    denominator <- n_successful
    power <- if (denominator > 0L) n_rejections / denominator else NA_real_
  } else {
    denominator <- trials
    power <- n_rejections / denominator
  }

  mcse <- if (is.finite(power) && denominator > 0L) {
    sqrt(power * (1 - power) / denominator)
  } else {
    NA_real_
  }

  list(
    power = power,
    mcse = mcse,
    n_trials = trials,
    n_successful_fits = n_successful,
    n_rejections = n_rejections,
    n_failed = n_failed,
    failure_rate = n_failed / trials,
    effective_trials = denominator,
    failure_handling = failure_handling
  )
}

.unadjusted_message <- function() {
  paste(
    "adjust_period = FALSE omits period effects from the analysis model.",
    "In a stepped-wedge trial exposure is confounded with calendar time, so an",
    "unadjusted analysis can attribute secular trends to the intervention and",
    "bias the noninferiority test. Use it only to illustrate that consequence."
  )
}

.warn_unadjusted <- function() {
  cond <- structure(
    class = c("swNIsim_unadjusted_warning", "warning", "condition"),
    list(message = .unadjusted_message(), call = NULL)
  )
  warning(cond)
}

.muffle_unadjusted <- function(expr) {
  withCallingHandlers(
    expr,
    swNIsim_unadjusted_warning = function(w) invokeRestart("muffleWarning")
  )
}

.resolve_fit_cp <- function(fit_cp, outcome, sdcp) {
  unset <- is.null(fit_cp) || (length(fit_cp) == 1L && is.na(fit_cp))
  if (identical(outcome, "continuous")) {
    if (!unset) {
      stop(
        "fit_cp applies to binary outcomes only. The continuous analysis of ",
        "cluster-period means absorbs a cluster-by-period effect in its residual variance.",
        call. = FALSE
      )
    }
    return(FALSE)
  }
  if (unset) return(sdcp > 0)
  if (!is.logical(fit_cp) || length(fit_cp) != 1L) {
    stop("fit_cp must be NULL, TRUE or FALSE.", call. = FALSE)
  }
  fit_cp
}

.correlation_summary <- function(outcome, sdc, sdcp, sdw) {
  s2 <- if (identical(outcome, "binary")) pi^2 / 3 else sdw^2
  total <- sdc^2 + sdcp^2 + s2
  cluster_var <- sdc^2 + sdcp^2
  list(
    scale = if (identical(outcome, "binary")) "latent logistic" else "outcome",
    structure = if (sdcp > 0) "nested exchangeable" else "exchangeable",
    within_period_icc = cluster_var / total,
    between_period_icc = sdc^2 / total,
    cluster_autocorrelation = if (cluster_var > 0) sdc^2 / cluster_var else NA_real_
  )
}
