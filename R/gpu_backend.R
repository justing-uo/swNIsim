.swnisim_state <- new.env(parent = emptyenv())

.torch_script_path <- function() {
  installed <- system.file("python", "swnisim_torch.py", package = "swNIsim")
  if (nzchar(installed) && file.exists(installed)) return(installed)

  # Fallback for pkgload / source()-based local testing
  candidates <- c(
    file.path("inst", "python", "swnisim_torch.py"),
    file.path("..", "inst", "python", "swnisim_torch.py")
  )
  for (cand in candidates) {
    if (file.exists(cand)) return(normalizePath(cand, winslash = "/", mustWork = TRUE))
  }
  stop("Could not locate inst/python/swnisim_torch.py in swNIsim.", call. = FALSE)
}

.candidate_python_bins <- function() {
  bins <- character(0L)

  env_py <- Sys.getenv("SWNISIM_PYTHON", unset = "")
  if (nzchar(env_py)) bins <- c(bins, env_py)

  retic_py <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(retic_py)) bins <- c(bins, retic_py)

  if (requireNamespace("reticulate", quietly = TRUE)) {
    if (isTRUE(tryCatch(reticulate::py_available(initialize = FALSE), error = function(e) FALSE))) {
      cfg_py <- tryCatch(as.character(reticulate::py_config()$python), error = function(e) "")
      if (nzchar(cfg_py)) bins <- c(bins, cfg_py)
    }
  }

  which_py <- Sys.which(c("python3", "python"))
  bins <- c(
    bins,
    unname(which_py[nzchar(which_py)]),
    "/usr/local/bin/python3",
    "/usr/bin/python3",
    "/opt/conda/bin/python3"
  )

  bins <- unique(bins[nzchar(bins)])
  bins[file.exists(bins)]
}

.parse_flat_probe_json <- function(txt) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::fromJSON(txt, simplifyVector = TRUE))
  }
  extract_bool <- function(key) {
    m <- regmatches(txt, regexpr(paste0('"', key, '"\\s*:\\s*(true|false)'), txt))
    if (!length(m) || !nzchar(m)) return(FALSE)
    grepl("true$", m)
  }
  extract_str <- function(key) {
    pat <- paste0('"', key, '"\\s*:\\s*"([^"]*)"')
    reg <- regexec(pat, txt)
    hit <- regmatches(txt, reg)[[1L]]
    if (length(hit) >= 2L) hit[2L] else ""
  }
  extract_int <- function(key) {
    pat <- paste0('"', key, '"\\s*:\\s*(-?[0-9]+)')
    reg <- regexec(pat, txt)
    hit <- regmatches(txt, reg)[[1L]]
    if (length(hit) >= 2L) as.integer(hit[2L]) else 0L
  }
  list(
    available = extract_bool("available"),
    torch_version = extract_str("torch_version"),
    cuda_available = extract_bool("cuda_available"),
    cuda_device_count = extract_int("cuda_device_count"),
    mps_available = extract_bool("mps_available"),
    device_name = extract_str("device_name"),
    recommended_device = extract_str("recommended_device"),
    cuda_kernel_supported = extract_bool("cuda_kernel_supported")
  )
}

.probe_single_python <- function(py_bin, script_path) {
  out <- tryCatch(
    suppressWarnings(system2(
      py_bin,
      args = c(shQuote(script_path), "--probe"),
      stdout = TRUE,
      stderr = FALSE
    )),
    error = function(e) character(0L)
  )
  status <- attr(out, "status")
  if (!is.null(status) && status != 0L) return(NULL)
  if (!length(out)) return(NULL)
  parsed <- tryCatch(
    .parse_flat_probe_json(paste(out, collapse = "\n")),
    error = function(e) NULL
  )
  if (is.null(parsed) || !isTRUE(parsed$available)) return(NULL)
  parsed$python <- py_bin
  parsed
}

.discover_torch_env <- function(refresh = FALSE) {
  if (!isTRUE(refresh) && exists("probe_info", envir = .swnisim_state, inherits = FALSE)) {
    return(get("probe_info", envir = .swnisim_state, inherits = FALSE))
  }

  script_path <- tryCatch(.torch_script_path(), error = function(e) "")
  bins <- if (nzchar(script_path)) .candidate_python_bins() else character(0L)

  # If explicit SWNISIM_PYTHON is set, only test that binary unless it fails? No:
  # honor SWNISIM_PYTHON strictly if set so users can pin an exact environment.
  explicit_py <- Sys.getenv("SWNISIM_PYTHON", unset = "")
  if (nzchar(explicit_py)) {
    bins <- if (file.exists(explicit_py)) explicit_py else character(0L)
  }

  best <- NULL
  for (py_bin in bins) {
    info <- .probe_single_python(py_bin, script_path)
    if (is.null(info)) next
    if (is.null(best)) best <- info
    if (isTRUE(info$cuda_available) || isTRUE(info$mps_available)) {
      best <- info
      break
    }
  }

  if (is.null(best)) {
    best <- list(
      available = FALSE,
      python = if (length(bins)) bins[1L] else "",
      torch_version = "",
      cuda_available = FALSE,
      cuda_device_count = 0L,
      mps_available = FALSE,
      device_name = "",
      recommended_device = "none",
      cuda_kernel_supported = FALSE
    )
  }

  has_reticulate <- requireNamespace("reticulate", quietly = TRUE)
  best$bridge <- if (!isTRUE(best$available)) {
    "none"
  } else if (has_reticulate) {
    "reticulate"
  } else {
    "python_cli"
  }

  if (isTRUE(refresh) && exists("py_mod", envir = .swnisim_state, inherits = FALSE)) {
    rm("py_mod", envir = .swnisim_state)
  }
  assign("probe_info", best, envir = .swnisim_state)
  best
}

.get_reticulate_module <- function(probe_info) {
  if (exists("py_mod", envir = .swnisim_state, inherits = FALSE)) {
    return(get("py_mod", envir = .swnisim_state, inherits = FALSE))
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)

  if (!isTRUE(tryCatch(reticulate::py_available(initialize = FALSE), error = function(e) FALSE))) {
    if (nzchar(probe_info$python)) {
      try(reticulate::use_python(probe_info$python, required = FALSE), silent = TRUE)
    }
  }

  py_dir <- dirname(.torch_script_path())
  mod <- tryCatch(
    reticulate::import_from_path("swnisim_torch", path = py_dir, convert = TRUE),
    error = function(e) NULL
  )
  if (!is.null(mod)) {
    # Verify this reticulate session actually sees torch (in case reticulate was
    # previously locked to a different Python without torch).
    ok <- isTRUE(tryCatch(mod$probe_torch()$available, error = function(e) FALSE))
    if (!ok) mod <- NULL
  }
  if (!is.null(mod)) {
    assign("py_mod", mod, envir = .swnisim_state)
  }
  mod
}

.to_json_minimal <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::toJSON(x, auto_unbox = TRUE, digits = 16, null = "null"))
  }
  encode_val <- function(v) {
    if (is.null(v)) return("null")
    if (is.logical(v) && length(v) == 1L) return(if (isTRUE(v)) "true" else "false")
    if (is.character(v) && length(v) == 1L) {
      esc <- gsub("\\\\", "\\\\\\\\", v)
      esc <- gsub('"', '\\\\"', esc)
      return(paste0('"', esc, '"'))
    }
    if (is.numeric(v) && length(v) == 1L && !inherits(v, "AsIs")) {
      if (is.integer(v)) return(sprintf("%d", v))
      return(sprintf("%.17g", v))
    }
    if (is.atomic(v) && !is.list(v)) {
      if (is.integer(v)) {
        return(paste0("[", paste(sprintf("%d", v), collapse = ","), "]"))
      }
      if (is.numeric(v)) {
        return(paste0("[", paste(sprintf("%.17g", v), collapse = ","), "]"))
      }
    }
    if (is.list(v)) {
      nms <- names(v)
      if (is.null(nms)) {
        items <- vapply(v, encode_val, character(1L))
        return(paste0("[", paste(items, collapse = ","), "]"))
      }
      pairs <- vapply(seq_along(v), function(i) {
        paste0('"', nms[i], '":', encode_val(v[[i]]))
      }, character(1L))
      return(paste0("{", paste(pairs, collapse = ","), "}"))
    }
    stop("Unsupported type in minimal JSON encoder.", call. = FALSE)
  }
  encode_val(x)
}

.parse_sim_response_json <- function(path, is_grid = FALSE) {
  txt <- paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::fromJSON(txt, simplifyVector = !is_grid, simplifyDataFrame = FALSE))
  }
  parse_one <- function(chunk) {
    get_int <- function(key) {
      reg <- regexec(paste0('"', key, '"\\s*:\\s*(-?[0-9]+)'), chunk)
      hit <- regmatches(chunk, reg)[[1L]]
      as.integer(hit[2L])
    }
    get_str <- function(key) {
      reg <- regexec(paste0('"', key, '"\\s*:\\s*"([^"]*)"'), chunk)
      hit <- regmatches(chunk, reg)[[1L]]
      if (length(hit) >= 2L) hit[2L] else ""
    }
    list(
      n_trials = get_int("n_trials"),
      n_rejections = get_int("n_rejections"),
      n_failed = get_int("n_failed"),
      device = get_str("device"),
      dtype = get_str("dtype")
    )
  }
  if (!is_grid) return(parse_one(txt))
  # Split top-level JSON array of flat objects
  objs <- regmatches(txt, gregexpr("\\{[^{}]*\\}", txt))[[1L]]
  lapply(objs, parse_one)
}

.call_torch_cli <- function(py_bin, payload, is_grid = FALSE) {
  req_file <- tempfile(pattern = "swnisim_req_", fileext = ".json")
  resp_file <- tempfile(pattern = "swnisim_resp_", fileext = ".json")
  on.exit(unlink(c(req_file, resp_file)), add = TRUE)

  writeLines(.to_json_minimal(payload), con = req_file, useBytes = TRUE)
  script_path <- .torch_script_path()
  err_out <- tryCatch(
    system2(
      py_bin,
      args = c(shQuote(script_path), "--run", shQuote(req_file), shQuote(resp_file)),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) structure(conditionMessage(e), status = 1L)
  )
  status <- attr(err_out, "status")
  if ((!is.null(status) && status != 0L) || !file.exists(resp_file)) {
    stop(
      "PyTorch simulation backend failed:\n",
      paste(err_out, collapse = "\n"),
      call. = FALSE
    )
  }
  .parse_sim_response_json(resp_file, is_grid = is_grid)
}

#' Check GPU and PyTorch backend availability
#'
#' Reports whether Python `torch`, CUDA/MPS GPU acceleration, `reticulate`, and
#' `lme4` are available in the current R session (including Google Colab).
#'
#' @param refresh Logical; if `TRUE`, re-probe Python environments instead of
#'   using cached session status.
#'
#' @return A list of class `sw_ni_gpu_status` describing available backends and
#'   hardware devices.
#' @export
sw_ni_gpu_status <- function(refresh = FALSE) {
  info <- .discover_torch_env(refresh = refresh)
  gpu_ok <- isTRUE(info$available) && (isTRUE(info$cuda_available) || isTRUE(info$mps_available))
  lme4_ok <- requireNamespace("lme4", quietly = TRUE)

  # Check if reticulate module actually loads when reticulate is present
  bridge <- info$bridge
  if (identical(bridge, "reticulate")) {
    mod <- .get_reticulate_module(info)
    if (is.null(mod)) bridge <- "python_cli"
  }

  auto_backend <- if (gpu_ok) {
    "gpu"
  } else if (isTRUE(info$available)) {
    "torch_cpu"
  } else if (lme4_ok) {
    "lme4"
  } else {
    "unavailable"
  }

  out <- list(
    torch_available = isTRUE(info$available),
    gpu_available = gpu_ok,
    cuda_available = isTRUE(info$cuda_available),
    cuda_device_count = as.integer(info$cuda_device_count),
    mps_available = isTRUE(info$mps_available),
    device_name = as.character(info$device_name),
    recommended_device = as.character(info$recommended_device),
    cuda_kernel_supported = isTRUE(info$cuda_kernel_supported),
    torch_version = as.character(info$torch_version),
    python = as.character(info$python),
    bridge = as.character(bridge),
    lme4_available = lme4_ok,
    default_backend = auto_backend
  )
  class(out) <- "sw_ni_gpu_status"
  out
}

#' @export
print.sw_ni_gpu_status <- function(x, ...) {
  cat("swNIsim Backend Status\n")
  cat("----------------------\n")
  cat("Default 'auto' backend : ", x$default_backend, "\n", sep = "")
  cat("GPU available          : ", x$gpu_available,
      if (nzchar(x$device_name)) paste0(" (", x$device_name, ")") else "", "\n", sep = "")
  cat("PyTorch available      : ", x$torch_available,
      if (nzchar(x$torch_version)) paste0(" (v", x$torch_version, ", bridge: ", x$bridge, ")") else "", "\n", sep = "")
  cat("Fused CUDA kernels     : ", isTRUE(x$cuda_kernel_supported), "\n", sep = "")
  if (nzchar(x$python)) {
    cat("Python executable      : ", x$python, "\n", sep = "")
  }
  cat("lme4 CPU available     : ", x$lme4_available, "\n", sep = "")
  if (!x$gpu_available) {
    cat("Note: In Google Colab, enable GPU via Runtime -> Change runtime type -> T4 GPU.\n")
  }
  invisible(x)
}

.resolve_execution_backend <- function(backend = c("auto", "gpu", "torch_cpu", "lme4"),
                                       device = NULL,
                                       cluster = NULL,
                                       lme4_reason = NULL,
                                       torch_reason = NULL,
                                       prefer_lme4 = FALSE) {
  backend <- match.arg(backend)
  if (!is.null(lme4_reason) && backend %in% c("gpu", "torch_cpu")) {
    stop(lme4_reason, " is only available with backend = 'lme4' or 'auto'.", call. = FALSE)
  }
  if (!is.null(torch_reason) && identical(backend, "lme4")) {
    stop(torch_reason, " require backend = 'gpu', 'torch_cpu' or 'auto'.", call. = FALSE)
  }
  if (identical(backend, "auto") && is.null(torch_reason) &&
      (!is.null(lme4_reason) || isTRUE(prefer_lme4))) {
    backend <- "lme4"
  }
  if (!is.null(torch_reason) && identical(backend, "auto")) {
    info <- .discover_torch_env(refresh = FALSE)
    if (!isTRUE(info$available)) {
      stop(torch_reason, " require Python with PyTorch ('torch').", call. = FALSE)
    }
    cluster <- NULL
  }
  if (!is.null(device)) {
    if (!is.character(device) || length(device) != 1L || is.na(device) || !nzchar(device)) {
      stop("device must be NULL or a non-empty character string such as 'cuda' or 'cpu'.", call. = FALSE)
    }
  }

  if (identical(backend, "lme4")) {
    if (!requireNamespace("lme4", quietly = TRUE)) {
      stop(
        "Package 'lme4' is required when backend = 'lme4'. ",
        "Install 'lme4' or use backend = 'auto' / 'gpu' / 'torch_cpu'.",
        call. = FALSE
      )
    }
    return(list(engine = "lme4", device = "cpu", probe = NULL))
  }

  info <- .discover_torch_env(refresh = FALSE)
  gpu_ok <- isTRUE(info$available) && (isTRUE(info$cuda_available) || isTRUE(info$mps_available))

  if (identical(backend, "gpu")) {
    if (!isTRUE(info$available)) {
      stop(
        "backend = 'gpu' requires Python with PyTorch ('torch') installed. ",
        "In Google Colab, PyTorch is pre-installed; otherwise set SWNISIM_PYTHON to a Python with torch.",
        call. = FALSE
      )
    }
    target_dev <- if (!is.null(device)) device else info$recommended_device
    if (!gpu_ok && (is.null(device) || grepl("^(cuda|mps)", target_dev))) {
      stop(
        "backend = 'gpu' requested, but no CUDA/MPS GPU was detected by PyTorch. ",
        "In Google Colab, select Runtime -> Change runtime type -> T4 GPU, or use backend = 'auto'.",
        call. = FALSE
      )
    }
    return(list(engine = "torch", device = target_dev, probe = info, mode = "gpu"))
  }

  if (identical(backend, "torch_cpu")) {
    if (!isTRUE(info$available)) {
      stop(
        "backend = 'torch_cpu' requires Python with PyTorch ('torch') installed.",
        call. = FALSE
      )
    }
    return(list(engine = "torch", device = "cpu", probe = info, mode = "torch_cpu"))
  }

  # backend == "auto"
  if (!is.null(cluster) && requireNamespace("lme4", quietly = TRUE)) {
    return(list(engine = "lme4", device = "cpu", probe = info, mode = "lme4"))
  }

  if (isTRUE(info$available)) {
    target_dev <- if (!is.null(device)) {
      device
    } else if (gpu_ok) {
      info$recommended_device
    } else {
      "cpu"
    }
    mode_label <- if (grepl("^(cuda|mps)", target_dev)) "gpu" else "torch_cpu"
    return(list(engine = "torch", device = target_dev, probe = info, mode = mode_label))
  }

  if (requireNamespace("lme4", quietly = TRUE)) {
    return(list(engine = "lme4", device = "cpu", probe = info, mode = "lme4"))
  }

  stop(
    "Neither Python 'torch' nor R package 'lme4' is available. ",
    "In Google Colab, PyTorch is pre-installed; locally, install 'lme4' or Python 'torch'.",
    call. = FALSE
  )
}

.default_batch_size <- function(outcome, n_cells, p_dim, device, user_batch_size = NULL, sdcp = 0) {
  if (!is.null(user_batch_size)) {
    .check_scalar_numeric(user_batch_size, "batch_size", lower = 1, integer = TRUE)
    return(as.integer(user_batch_size))
  }
  is_gpu <- grepl("^(cuda|mps)", as.character(device))
  if (sdcp > 0) {
    if (identical(outcome, "continuous")) {
      return(if (is_gpu) 15000L else 1000L)
    } else {
      return(if (is_gpu) 10000L else 250L)
    }
  }
  if (identical(outcome, "continuous")) {
    base <- if (is_gpu) 40000L else 5000L
    scale <- max(1, (n_cells / 300) * (p_dim / 12))
    max(500L, as.integer(round(base / scale)))
  } else {
    base <- if (is_gpu) 20000L else 2000L
    scale <- max(1, (n_cells / 1000) * ((p_dim / 32)^2))
    max(250L, as.integer(round(base / scale)))
  }
}

.build_torch_scenario_payload <- function(outcome,
                                          trials,
                                          seed,
                                          design_obj,
                                          K_vec,
                                          period_effects,
                                          adjust_period,
                                          b1,
                                          mu,
                                          sdw,
                                          sdc,
                                          theta,
                                          margin,
                                          sig.level,
                                          REML,
                                          batch_size,
                                          device,
                                          sdcp = 0,
                                          rho = 0) {
  b_sz <- .default_batch_size(
    outcome = outcome,
    n_cells = design_obj$n_cells,
    p_dim = if (isTRUE(adjust_period)) design_obj$Tp + 1L else 2L,
    device = device,
    user_batch_size = batch_size,
    sdcp = sdcp
  )
  list(
    outcome = as.character(outcome),
    trials = as.integer(trials),
    seed = as.integer(seed),
    I_clusters = as.integer(design_obj$I),
    Tp = as.integer(design_obj$Tp),
    K_vec = I(as.numeric(K_vec)),
    delayed_tx_vec = I(as.numeric(design_obj$data_base$delayed_tx)),
    period_effects_vec = I(as.numeric(period_effects)),
    adjust_period = isTRUE(adjust_period),
    b1 = as.numeric(b1),
    mu = as.numeric(mu),
    sdw = as.numeric(sdw),
    sdc = as.numeric(sdc),
    sdcp = as.numeric(sdcp),
    rho = as.numeric(rho),
    theta = as.numeric(theta),
    margin = as.numeric(margin),
    sig_level = as.numeric(sig.level),
    reml = isTRUE(REML),
    batch_size = as.integer(b_sz),
    device = as.character(device)
  )
}

.run_power_torch <- function(payload, probe_info) {
  mod <- .get_reticulate_module(probe_info)
  if (!is.null(mod)) {
    res <- mod$simulate_power_torch(
      outcome = payload$outcome,
      trials = payload$trials,
      seed = payload$seed,
      I_clusters = payload$I_clusters,
      Tp = payload$Tp,
      K_vec = as.list(as.numeric(payload$K_vec)),
      delayed_tx_vec = as.list(as.numeric(payload$delayed_tx_vec)),
      period_effects_vec = as.list(as.numeric(payload$period_effects_vec)),
      adjust_period = payload$adjust_period,
      b1 = payload$b1,
      mu = payload$mu,
      sdw = payload$sdw,
      sdc = payload$sdc,
      sdcp = if (is.null(payload$sdcp)) 0 else payload$sdcp,
      rho = if (is.null(payload$rho)) 0 else payload$rho,
      theta = payload$theta,
      margin = payload$margin,
      sig_level = payload$sig_level,
      reml = payload$reml,
      batch_size = payload$batch_size,
      device = payload$device
    )
    return(res)
  }

  cli_payload <- c(list(`_fn` = "simulate_power_torch"), payload)
  .call_torch_cli(probe_info$python, cli_payload, is_grid = FALSE)
}

.run_grid_torch <- function(payload_list, probe_info) {
  mod <- .get_reticulate_module(probe_info)
  if (!is.null(mod)) {
    return(lapply(payload_list, function(p) .run_power_torch(p, probe_info)))
  }
  # Single Python invocation for the entire grid when reticulate is absent!
  cli_payload <- list(`_fn` = "simulate_grid_torch", scenarios = payload_list)
  .call_torch_cli(probe_info$python, cli_payload, is_grid = TRUE)
}

.compute_power_summary_from_counts <- function(n_rejections,
                                               n_failed,
                                               trials,
                                               failure_handling = c("nonrejection", "exclude", "error")) {
  failure_handling <- match.arg(failure_handling)
  n_failed <- as.integer(n_failed)
  n_rejections <- as.integer(n_rejections)
  trials <- as.integer(trials)
  n_successful <- trials - n_failed

  if (failure_handling == "error" && n_failed > 0L) {
    stop(n_failed, " model fit(s) failed. Set failure_handling to 'nonrejection' or 'exclude'.", call. = FALSE)
  }

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

# Which engine a scenario needs. lme4_reason / torch_reason are NULL or a short
# description of the setting that only that engine supports.
.engine_requirements <- function(outcome, sdcp, rho, nAGQ, fit_cp_user) {
  lme4_reason <- NULL
  if (identical(outcome, "binary") && nAGQ != 0L) lme4_reason <- "nAGQ != 0"
  if (identical(outcome, "binary") && isTRUE(fit_cp_user)) lme4_reason <- "an explicit fit_cp"
  torch_reason <- if (sdcp > 0 && rho != 0) "AR(1) cluster-by-period effects (sdcp > 0, rho != 0)" else NULL
  if (!is.null(lme4_reason) && !is.null(torch_reason)) {
    stop(lme4_reason, " requires the lme4 engine, but ", torch_reason,
         " requires a PyTorch backend.", call. = FALSE)
  }
  list(
    lme4_reason = lme4_reason,
    torch_reason = torch_reason,
    prefer_lme4 = sdcp > 0 && rho == 0
  )
}
