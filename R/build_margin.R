#' Convert a clinically specified non-inferiority margin to the model scale
#'
#' Translates a clinically specified non-inferiority threshold into the margin
#' used by [sw_ni_power()]: a mean difference for continuous outcomes, or a
#' log odds ratio for binary outcomes. Outcomes are oriented so that larger
#' values are favourable.
#'
#' The conversion route must be chosen explicitly with `method`, and only the
#' inputs that route uses may be supplied. There are no default reference
#' values or preserved fractions.
#'
#' \describe{
#'   \item{`"preserve_control"`}{The intervention must retain at least a
#'     fraction `f1` of the control response level: the minimum acceptable
#'     value is `f1 * p1` (binary) or `f1 * mu` (continuous). For continuous
#'     outcomes this is meaningful only on a scale with a true zero, so `mu`
#'     must be positive.}
#'   \item{`"preserve_benefit"`}{The intervention must retain at least a
#'     fraction `f2` of the control's established benefit over a benchmark
#'     such as placebo: the minimum acceptable value is `p0 + f2 * (p1 - p0)`
#'     or `mu0 + f2 * (mu - mu0)`. Justifying this choice requires relevant
#'     historical evidence for the benchmark.}
#'   \item{`"threshold"`}{The minimum acceptable value is given directly as
#'     `p_min` (binary) or `mu_min` (continuous).}
#'   \item{`"direct"`}{The margin is already on the model scale and is
#'     supplied as `margin`. For binary outcomes `p1` may also be given so that
#'     the reference log odds `b1` is returned.}
#' }
#'
#' For binary outcomes, `p1` and the minimum acceptable probability are
#' cluster-conditional probabilities (for a cluster whose random intercept is
#' zero), and the resulting margin is a cluster-conditional log odds ratio,
#' `qlogis(p1) - qlogis(p_min)`. This conversion needs no variance component,
#' but the result is not a population-average risk-difference margin.
#' Translating a population-average target to the conditional scale would also
#' require the between-cluster variance.
#'
#' Neither the choice of fraction nor the resulting value is evidence that a
#' margin is clinically acceptable.
#'
#' @param outcome Either `"binary"` or `"continuous"`.
#' @param method Conversion route: `"preserve_control"`, `"preserve_benefit"`,
#'   `"threshold"` or `"direct"`. Required.
#' @param p1 Binary: control (reference) success probability, strictly between
#'   0 and 1, on the cluster-conditional scale.
#' @param mu Continuous: control (reference) mean.
#' @param f1 Preserved fraction of the control response level, strictly
#'   between 0 and 1. Used only by `"preserve_control"`.
#' @param p0 Binary: benchmark (for example placebo) success probability,
#'   at least 0 and below `p1`. Used only by `"preserve_benefit"`.
#' @param mu0 Continuous: benchmark (for example placebo) mean, below `mu`.
#'   Used only by `"preserve_benefit"`.
#' @param f2 Preserved fraction of the control's benefit over the benchmark,
#'   strictly between 0 and 1. Used only by `"preserve_benefit"`.
#' @param p_min Binary: minimum acceptable success probability, strictly
#'   between 0 and `p1`. Used only by `"threshold"`.
#' @param mu_min Continuous: minimum acceptable mean, below `mu`. Used only by
#'   `"threshold"`.
#' @param margin A positive margin already on the model scale. Used only by
#'   `"direct"`.
#'
#' @return A list with elements `margin` (on the model scale), `b1` (binary
#'   reference log odds, or `NULL`), `mu` (continuous reference mean, or
#'   `NULL`), `outcome`, `method`, `reference` (the control value), and
#'   `minimum_acceptable` (the threshold implied on the outcome scale, or `NA`
#'   for `"direct"`).
#'
#' @examples
#' # Binary: keep at least 80% of a 30% control success rate
#' build_margin("binary", method = "preserve_control", p1 = 0.30, f1 = 0.80)
#'
#' # Binary: a clinically specified minimum of 24%
#' build_margin("binary", method = "threshold", p1 = 0.30, p_min = 0.24)
#'
#' # Continuous: keep at least half of the control's benefit over placebo
#' build_margin("continuous", method = "preserve_benefit",
#'              mu = 21, mu0 = 15, f2 = 0.5)
#' @export
build_margin <- function(outcome = c("binary", "continuous"),
                         method,
                         p1 = NULL,
                         mu = NULL,
                         f1 = NULL,
                         p0 = NULL,
                         mu0 = NULL,
                         f2 = NULL,
                         p_min = NULL,
                         mu_min = NULL,
                         margin = NULL) {
  outcome <- match.arg(outcome)
  methods <- c("preserve_control", "preserve_benefit", "threshold", "direct")
  if (missing(method) || is.null(method)) {
    stop(
      "method must be specified: one of ",
      paste0("\"", methods, "\"", collapse = ", "), ".",
      call. = FALSE
    )
  }
  if (!is.character(method) || length(method) != 1L || !method %in% methods) {
    stop(
      "method must be one of ",
      paste0("\"", methods, "\"", collapse = ", "), ".",
      call. = FALSE
    )
  }

  supplied <- list(
    p1 = p1, mu = mu, f1 = f1, p0 = p0, mu0 = mu0, f2 = f2,
    p_min = p_min, mu_min = mu_min, margin = margin
  )
  needed <- .margin_inputs(outcome, method)
  allowed <- c(needed$required, needed$optional)
  extra <- setdiff(names(supplied)[!vapply(supplied, is.null, logical(1L))], allowed)
  if (length(extra) > 0L) {
    stop(
      "Argument(s) ", paste(extra, collapse = ", "),
      " are not used for outcome = \"", outcome, "\" with method = \"", method,
      "\". Supply only: ", paste(allowed, collapse = ", "), ".",
      call. = FALSE
    )
  }
  missing_args <- needed$required[vapply(supplied[needed$required], is.null, logical(1L))]
  if (length(missing_args) > 0L) {
    stop(
      "method = \"", method, "\" for a ", outcome, " outcome requires: ",
      paste(missing_args, collapse = ", "), ".",
      call. = FALSE
    )
  }

  for (nm in names(supplied)) {
    if (!is.null(supplied[[nm]])) .check_scalar_numeric(supplied[[nm]], nm)
  }

  b1 <- NULL
  minimum <- NA_real_

  if (identical(outcome, "binary")) {
    if (!is.null(p1)) {
      .check_open_unit(p1, "p1")
      b1 <- stats::qlogis(p1)
    }
    if (identical(method, "preserve_control")) {
      .check_open_unit(f1, "f1")
      minimum <- f1 * p1
    } else if (identical(method, "preserve_benefit")) {
      if (p0 < 0 || p0 >= p1) {
        stop("p0 must be at least 0 and below p1, so that the control has a benefit to preserve.", call. = FALSE)
      }
      .check_open_unit(f2, "f2")
      minimum <- p0 + f2 * (p1 - p0)
    } else if (identical(method, "threshold")) {
      if (p_min <= 0 || p_min >= p1) {
        stop("p_min must be above 0 and below p1.", call. = FALSE)
      }
      minimum <- p_min
    }
    if (!identical(method, "direct")) {
      margin <- b1 - stats::qlogis(minimum)
    }
    reference <- if (is.null(p1)) NA_real_ else p1
  } else {
    if (identical(method, "preserve_control")) {
      if (mu <= 0) {
        stop(
          "method = \"preserve_control\" needs mu > 0: a fraction of the control mean ",
          "is meaningful only on a scale with a true zero. Use \"threshold\" or ",
          "\"preserve_benefit\" instead.",
          call. = FALSE
        )
      }
      .check_open_unit(f1, "f1")
      minimum <- f1 * mu
    } else if (identical(method, "preserve_benefit")) {
      if (mu0 >= mu) {
        stop("mu0 must be below mu, so that the control has a benefit to preserve.", call. = FALSE)
      }
      .check_open_unit(f2, "f2")
      minimum <- mu0 + f2 * (mu - mu0)
    } else if (identical(method, "threshold")) {
      if (mu_min >= mu) {
        stop("mu_min must be below mu.", call. = FALSE)
      }
      minimum <- mu_min
    }
    if (!identical(method, "direct")) {
      margin <- mu - minimum
    }
    reference <- if (is.null(mu)) NA_real_ else mu
  }

  if (!is.finite(margin) || margin <= 0) {
    stop("The resulting margin must be a positive finite number.", call. = FALSE)
  }

  list(
    margin = margin,
    b1 = b1,
    mu = if (identical(outcome, "continuous")) mu else NULL,
    outcome = outcome,
    method = method,
    reference = reference,
    minimum_acceptable = minimum
  )
}

.margin_inputs <- function(outcome, method) {
  if (identical(outcome, "binary")) {
    switch(method,
      preserve_control = list(required = c("p1", "f1"), optional = character()),
      preserve_benefit = list(required = c("p1", "p0", "f2"), optional = character()),
      threshold = list(required = c("p1", "p_min"), optional = character()),
      direct = list(required = "margin", optional = "p1")
    )
  } else {
    switch(method,
      preserve_control = list(required = c("mu", "f1"), optional = character()),
      preserve_benefit = list(required = c("mu", "mu0", "f2"), optional = character()),
      threshold = list(required = c("mu", "mu_min"), optional = character()),
      direct = list(required = "margin", optional = "mu")
    )
  }
}

.check_open_unit <- function(x, name) {
  if (x <= 0 || x >= 1) {
    stop(name, " must be strictly between 0 and 1.", call. = FALSE)
  }
  invisible(TRUE)
}
