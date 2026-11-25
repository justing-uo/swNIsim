# Install swNIsim 1.3 locally or in Google Colab.
# lme4 is needed for the CPU reference engine (and for nAGQ, fit_cp and nested
# exchangeable correlation). reticulate is optional and speeds up PyTorch calls.
install.packages("lme4")
# install.packages("reticulate")

# Adjust this path to where the tarball was downloaded.
pkg_file <- "~/Downloads/swNIsim_1.3.tar.gz"
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
