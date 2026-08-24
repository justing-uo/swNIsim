# swNIsim 0.1.2

- Added fixed categorical period adjustment to binary and continuous analysis models.
- Added user-specified secular `period_effects` to both data-generating models.
- Added unequal `clusters_per_sequence` support.
- Added unequal `K` support at cluster or cluster-period level, including matrices.
- Added precision weights for continuous cluster-period means when `K` varies.
- Added conservative failed-fit handling and Monte Carlo standard errors.
- Added actual total clusters, cluster-period cells, and observations to detailed output.
- Extended the Shiny app and grid runner for the new arguments.
