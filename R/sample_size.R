#' Find the smallest design that meets a power target
#'
#' Searches over the common cluster-period size `K`, or over the number of
#' clusters per crossover sequence, for the smallest value whose simulated
#' power reaches `target_power`. All other design and model settings are passed
#' unchanged to [sw_ni_power()].
#'
#' By default the search is a bisection between `lower` and `upper`, which
#' assumes that power increases with the searched quantity. It evaluates about
#' `log2(upper - lower) + 2` candidates. Supply `values` instead to evaluate
#' every listed candidate, which also gives the power curve.
#'
#' Every candidate uses the same `seed` and `trials`. Because power is
#' estimated, a candidate whose estimate lies within two Monte Carlo SEs of the
#' target could fall on either side of it; the result then sets
#' `near_target = TRUE`, and the answer should be confirmed with more `trials`
#' (for example by calling [sw_ni_power()] for the solution and the value just
#' below it).
#'
#' @param target_power Target power, strictly between 0 and 1.
#' @param search `"K"` to search over the common cluster-period size, or
#'   `"clusters"` to search over the number of clusters per sequence (a
#'   balanced allocation).
#' @param lower,upper Integer bounds for the bisection search. Defaults are
#'   2 and 200 for `"K"`, and 1 and 50 for `"clusters"`. Ignored when `values`
#'   is supplied.
#' @param values Optional vector of positive integers to evaluate in full
#'   instead of bisecting.
#' @param seed Random-number seed used for every candidate.
#' @param trials Number of simulated trials per candidate.
#' @param ... Other arguments to [sw_ni_power()], such as `outcome`, `Tp`, `K`
#'   or `no_clusters_per_step` (whichever is not searched), `mu`, `sdw`, `sdc`,
#'   `b1`, `theta` and `margin`. Do not supply the searched quantity, `seed`,
#'   `trials` or `return_details`. When searching over clusters, `K` must be a
#'   single number.
#'
#' @return An object of class `sw_ni_sample_size`: a list with the smallest
#'   value meeting the target (`value`, or `NA` if none does), its estimated
#'   `power` and `mcse`, `reached`, `near_target`, `monotone` (whether the
#'   evaluated power never falls below the target above the solution),
#'   `search`, `target_power`, `trials`, `seed`, and `evaluated`, a data frame
#'   of every candidate simulated.
#'
#' @examples
#' \dontrun{
#' # TheraPPP-based illustration: smallest number of patients per
#' # cluster-period for 90% power with 16 ICUs over nine periods
#' sw_ni_sample_size(
#'   target_power = 0.90, search = "K", lower = 2, upper = 20,
#'   outcome = "continuous", Tp = 9, no_clusters_per_step = 2,
#'   mu = 21, sdw = 10, sdc = 4.2, theta = 0, margin = 4.2,
#'   trials = 10000
#' )
#' }
#' @export
sw_ni_sample_size <- function(target_power = 0.9,
                              search = c("K", "clusters"),
                              lower = NULL,
                              upper = NULL,
                              values = NULL,
                              seed = 2026,
                              trials = 10000,
                              ...) {
  search <- match.arg(search)
  .check_scalar_numeric(target_power, "target_power")
  if (target_power <= 0 || target_power >= 1) {
    stop("target_power must be strictly between 0 and 1.", call. = FALSE)
  }
  .check_scalar_numeric(seed, "seed", integer = TRUE)
  .check_scalar_numeric(trials, "trials", lower = 1, integer = TRUE)

  args <- list(...)
  if (length(args) > 0L && (is.null(names(args)) || any(!nzchar(names(args))))) {
    stop("All arguments passed through ... must be named.", call. = FALSE)
  }
  searched <- if (identical(search, "K")) "K" else c("no_clusters_per_step", "clusters_per_sequence")
  forbidden <- intersect(names(args), c(searched, "seed", "trials", "return_details"))
  if (length(forbidden) > 0L) {
    stop(
      "Do not supply ", paste(forbidden, collapse = ", "),
      " when searching over ", search, ".",
      call. = FALSE
    )
  }
  if (identical(search, "clusters") && !is.null(args$K) &&
      (is.matrix(args$K) || length(args$K) != 1L)) {
    stop("When searching over clusters, K must be a single number.", call. = FALSE)
  }

  if (is.null(values)) {
    if (is.null(lower)) lower <- if (identical(search, "K")) 2L else 1L
    if (is.null(upper)) upper <- if (identical(search, "K")) 200L else 50L
    .check_scalar_numeric(lower, "lower", lower = 1, integer = TRUE)
    .check_scalar_numeric(upper, "upper", lower = 1, integer = TRUE)
    if (upper <= lower) stop("upper must be greater than lower.", call. = FALSE)
    lower <- as.integer(lower)
    upper <- as.integer(upper)
  } else {
    .check_positive_integer_vector(values, "values")
    values <- sort(unique(as.integer(values)))
  }

  if (isFALSE(args$adjust_period)) .warn_unadjusted()

  cache <- new.env(parent = emptyenv())
  evaluate <- function(v) {
    key <- as.character(v)
    if (!is.null(cache[[key]])) return(cache[[key]])
    call_args <- c(args, list(seed = seed, trials = trials, return_details = TRUE))
    if (identical(search, "K")) {
      call_args$K <- v
    } else {
      call_args$no_clusters_per_step <- v
    }
    out <- .muffle_unadjusted(do.call(sw_ni_power, call_args))
    row <- data.frame(
      value = v,
      power = out$power,
      mcse = out$mcse,
      n_failed = out$n_failed,
      failure_rate = out$failure_rate,
      total_clusters = out$design$total_clusters,
      total_observations = out$design$total_observations
    )
    assign(key, row, envir = cache)
    row
  }
  power_at <- function(v) evaluate(v)$power

  if (is.null(values)) {
    solution <- .bisect_smallest(power_at, lower, upper, target_power)
  } else {
    pw <- vapply(values, power_at, numeric(1L))
    meets <- values[!is.na(pw) & pw >= target_power]
    solution <- if (length(meets) > 0L) min(meets) else NA_integer_
  }

  evaluated <- do.call(rbind, mget(ls(cache), envir = cache))
  evaluated <- evaluated[order(evaluated$value), , drop = FALSE]
  rownames(evaluated) <- NULL
  evaluated$meets_target <- !is.na(evaluated$power) & evaluated$power >= target_power

  reached <- !is.na(solution)
  sol_row <- if (reached) evaluated[evaluated$value == solution, , drop = FALSE] else NULL
  near <- function(r) {
    !is.null(r) && nrow(r) == 1L && is.finite(r$mcse) &&
      abs(r$power - target_power) < 2 * r$mcse
  }
  below_row <- if (reached) {
    b <- evaluated[evaluated$value < solution, , drop = FALSE]
    if (nrow(b) > 0L) b[nrow(b), , drop = FALSE] else NULL
  } else {
    NULL
  }
  monotone <- if (reached) {
    all(evaluated$meets_target[evaluated$value >= solution])
  } else {
    TRUE
  }

  structure(
    list(
      value = if (reached) as.integer(solution) else NA_integer_,
      power = if (reached) sol_row$power else NA_real_,
      mcse = if (reached) sol_row$mcse else NA_real_,
      reached = reached,
      near_target = reached && (near(sol_row) || near(below_row)),
      monotone = monotone,
      search = search,
      target_power = target_power,
      trials = as.integer(trials),
      seed = seed,
      evaluated = evaluated
    ),
    class = "sw_ni_sample_size"
  )
}

.bisect_smallest <- function(power_at, lower, upper, target) {
  p_upper <- power_at(upper)
  if (is.na(p_upper) || p_upper < target) return(NA_integer_)
  p_lower <- power_at(lower)
  if (!is.na(p_lower) && p_lower >= target) return(as.integer(lower))
  lo <- lower
  hi <- upper
  while (hi - lo > 1L) {
    mid <- lo + (hi - lo) %/% 2L
    p_mid <- power_at(mid)
    if (!is.na(p_mid) && p_mid >= target) hi <- mid else lo <- mid
  }
  as.integer(hi)
}

#' @export
print.sw_ni_sample_size <- function(x, ...) {
  label <- if (identical(x$search, "K")) "K (patients per cluster-period)" else "clusters per sequence"
  short <- if (identical(x$search, "K")) "K" else "clusters per sequence"
  fmt <- function(p) formatC(p, format = "f", digits = 3)
  target <- paste0(formatC(100 * x$target_power, format = "fg"), "%")

  if (!x$reached) {
    top <- max(x$evaluated$value)
    cat("No ", label, " up to ", top, " reached ", target, " power.\n", sep = "")
    r <- x$evaluated[x$evaluated$value == top, ]
    cat("  Power at ", short, " = ", top, ": ", fmt(r$power),
        " (Monte Carlo SE ", fmt(r$mcse), ")\n", sep = "")
  } else {
    r <- x$evaluated[x$evaluated$value == x$value, ]
    cat("Smallest ", label, " meeting ", target, " power: ", x$value, "\n", sep = "")
    cat("  Estimated power ", fmt(r$power), " (Monte Carlo SE ", fmt(r$mcse),
        "); total observations ", format(r$total_observations, big.mark = ","), "\n", sep = "")
    below <- x$evaluated[x$evaluated$value < x$value, , drop = FALSE]
    if (nrow(below) > 0L) {
      b <- below[nrow(below), ]
      cat("  At ", short, " = ", b$value, ": power ", fmt(b$power),
          " (Monte Carlo SE ", fmt(b$mcse), ")\n", sep = "")
    }
  }
  cat("Based on ", format(x$trials, big.mark = ","), " simulated trials per candidate (seed ",
      x$seed, ").\n", sep = "")
  if (isTRUE(x$near_target)) {
    cat("Note: an estimate is within two Monte Carlo SEs of the target; confirm with more trials.\n")
  }
  if (!isTRUE(x$monotone)) {
    cat("Note: a larger candidate fell below the target; Monte Carlo noise may affect the answer.\n")
  }
  if (any(x$evaluated$n_failed > 0L)) {
    cat("Note: some model fits failed; see $evaluated.\n")
  }
  invisible(x)
}
