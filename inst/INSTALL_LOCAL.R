# Install swNIsim 0.2.0 locally or in Google Colab.
# Optional: install reticulate for in-process PyTorch GPU calls, or lme4 for CPU reference fits.
# install.packages(c("reticulate", "lme4"))

pkg_file <- "swNIsim_0.2.0.tar.gz"
install.packages(pkg_file, repos = NULL, type = "source")

library(swNIsim)
print(sw_ni_gpu_status())

# Quick continuous smoke test (uses GPU automatically when available).
sw_ni_power(
  outcome = "continuous",
  trials = 100,
  Tp = 5,
  no_clusters_per_step = 4,
  K = 20,
  return_details = TRUE
)
