# Google Colab Quickstart for swNIsim 0.2.0
# Enable T4/L4/A100 GPU in Colab: Runtime -> Change runtime type -> T4 GPU

library(swNIsim)
print(sw_ni_gpu_status())

# Single scenario (100,000 trials on GPU)
res <- sw_ni_power(
  outcome = "continuous",
  trials = 100000,
  Tp = 9,
  no_clusters_per_step = 2,
  K = 5,
  mu = 21,
  sdw = 10,
  sdc = 4.2,
  margin = 4.2,
  REML = TRUE,
  backend = "auto",
  return_details = TRUE
)
print(res[c("power", "mcse", "backend", "device")])
