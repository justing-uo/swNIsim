.sw_ni_simulate_binary_once <- function(design_obj,
                                        K_vec,
                                        period_effects,
                                        adjust_period = TRUE,
                                        b1 = 0.5,
                                        sdc = 1,
                                        theta = 0,
                                        margin = 0.25,
                                        sig.level = 0.025,
                                        control = NULL) {
  if (is.null(control)) {
    control <- lme4::glmerControl(check.conv.singular = "ignore")
  }

  data_sim <- design_obj$data_base
  data_sim$K <- K_vec

  alpha <- stats::rnorm(design_obj$I, mean = 0, sd = sdc)
  eta <- b1 + period_effects[data_sim$time] +
    alpha[data_sim$cluster_id] + theta * data_sim$delayed_tx
  p <- stats::plogis(eta)

  data_sim$success <- stats::rbinom(design_obj$n_cells, size = data_sim$K, prob = p)
  data_sim$fail <- data_sim$K - data_sim$success

  model_formula <- if (isTRUE(adjust_period)) {
    stats::as.formula("cbind(success, fail) ~ period + delayed_tx + (1 | cluster)")
  } else {
    stats::as.formula("cbind(success, fail) ~ delayed_tx + (1 | cluster)")
  }

  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      lme4::glmer(
        model_formula,
        data = data_sim,
        family = stats::binomial(),
        nAGQ = 0,
        control = control
      )
    )),
    error = function(e) NULL
  )

  if (is.null(fit)) return(NA_integer_)

  fe <- lme4::fixef(fit)
  if (!"delayed_tx" %in% names(fe)) return(NA_integer_)

  V <- tryCatch(as.matrix(stats::vcov(fit)), error = function(e) NULL)
  if (is.null(V) || !"delayed_tx" %in% rownames(V)) return(NA_integer_)

  coef_diff <- unname(fe["delayed_tx"])
  coef_se <- sqrt(V["delayed_tx", "delayed_tx"])
  if (!is.finite(coef_se) || coef_se <= 0) return(NA_integer_)

  z_NI <- (coef_diff + margin) / coef_se
  p_value <- 1 - stats::pnorm(z_NI)
  as.integer(p_value < sig.level)
}

.sw_ni_simulate_continuous_once <- function(design_obj,
                                            K_vec,
                                            period_effects,
                                            adjust_period = TRUE,
                                            mu = 5,
                                            sdw = 2,
                                            sdc = 1,
                                            theta = 0,
                                            margin = 0.15,
                                            sig.level = 0.025,
                                            REML = TRUE,
                                            control = NULL) {
  if (is.null(control)) {
    control <- lme4::lmerControl(check.conv.singular = "ignore")
  }

  data_sim <- design_obj$data_base
  data_sim$K <- K_vec
  alpha <- stats::rnorm(design_obj$I, mean = 0, sd = sdc)

  data_sim$mu_ij <- stats::rnorm(
    design_obj$n_cells,
    mean = mu + period_effects[data_sim$time] +
      alpha[data_sim$cluster_id] + theta * data_sim$delayed_tx,
    sd = sdw / sqrt(data_sim$K)
  )

  model_formula <- if (isTRUE(adjust_period)) {
    stats::as.formula("mu_ij ~ period + delayed_tx + (1 | cluster)")
  } else {
    stats::as.formula("mu_ij ~ delayed_tx + (1 | cluster)")
  }

  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      lme4::lmer(
        model_formula,
        data = data_sim,
        weights = data_sim$K,
        REML = REML,
        control = control
      )
    )),
    error = function(e) NULL
  )

  if (is.null(fit)) return(NA_integer_)

  fe <- lme4::fixef(fit)
  if (!"delayed_tx" %in% names(fe)) return(NA_integer_)

  V <- tryCatch(as.matrix(stats::vcov(fit)), error = function(e) NULL)
  if (is.null(V) || !"delayed_tx" %in% rownames(V)) return(NA_integer_)

  coef_diff <- unname(fe["delayed_tx"])
  coef_se <- sqrt(V["delayed_tx", "delayed_tx"])
  if (!is.finite(coef_se) || coef_se <= 0) return(NA_integer_)

  z_NI <- (coef_diff + margin) / coef_se
  p_value <- 1 - stats::pnorm(z_NI)
  as.integer(p_value < sig.level)
}
