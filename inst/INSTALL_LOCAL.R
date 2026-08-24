# Install the required dependency first.
install.packages("lme4")

# Adjust this path to where the tarball was downloaded.
pkg_file <- "~/Downloads/swNIsim_0.1.2.tar.gz"
install.packages(pkg_file, repos = NULL, type = "source")

library(swNIsim)

# Quick continuous smoke test.
sw_ni_power(
  outcome = "continuous",
  trials = 10,
  Tp = 5,
  no_clusters_per_step = 4,
  K = 20,
  return_details = TRUE
)
