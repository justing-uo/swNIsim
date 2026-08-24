#' Launch an interactive Shiny app
#'
#' Opens a Shiny interface for a single binary or continuous simulation
#' scenario. Comma-separated vectors can be entered for unequal sequence sizes,
#' cluster sizes, delayed effects, and secular period effects.
#'
#' @return A Shiny application object.
#' @export
run_sw_ni_app <- function() {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("Package 'shiny' is required. Install it with install.packages('shiny').", call. = FALSE)
  }

  ui <- shiny::fluidPage(
    shiny::titlePanel("Stepped-Wedge Non-Inferiority Power Simulation"),
    shiny::sidebarLayout(
      shiny::sidebarPanel(
        shiny::selectInput("outcome", "Outcome type", c("binary", "continuous")),
        shiny::numericInput("seed", "Seed", 2026, min = 1, step = 1),
        shiny::numericInput("trials", "Simulated trials", 1000, min = 1, step = 100),
        shiny::numericInput("Tp", "Time periods (Tp)", 5, min = 2, step = 1),
        shiny::numericInput("balanced_clusters", "Clusters per sequence (balanced default)", 4, min = 1, step = 1),
        shiny::textInput("clusters_per_sequence", "Unequal clusters by sequence (optional)", ""),
        shiny::textInput("K", "K: scalar or comma-separated cluster/cell sizes", "20"),
        shiny::textInput("period_effects", "Period effects (optional)", ""),
        shiny::checkboxInput("adjust_period", "Adjust for categorical period", TRUE),
        shiny::textInput("delay", "Post-crossover exposure values (optional)", ""),
        shiny::numericInput("sdc", "Cluster SD (sdc)", 1, min = 0, step = 0.1),
        shiny::numericInput("theta", "True treatment effect (theta)", 0, step = 0.01),
        shiny::numericInput("margin", "Non-inferiority margin", 0.25, min = 0, step = 0.01),
        shiny::numericInput("sig_level", "One-sided alpha", 0.025, min = 0, max = 1, step = 0.001),
        shiny::conditionalPanel(
          "input.outcome == 'binary'",
          shiny::numericInput("b1", "Baseline log-odds (b1)", 0.5, step = 0.1)
        ),
        shiny::conditionalPanel(
          "input.outcome == 'continuous'",
          shiny::numericInput("mu", "Baseline mean (mu)", 5, step = 0.1),
          shiny::numericInput("sdw", "Within-cluster SD (sdw)", 2, min = 0, step = 0.1),
          shiny::checkboxInput("REML", "Use REML", TRUE)
        ),
        shiny::selectInput(
          "failure_handling",
          "Failed-fit handling",
          c("Count as non-rejection" = "nonrejection", "Exclude" = "exclude", "Stop" = "error")
        ),
        shiny::checkboxInput("parallel", "Parallelize replicates", FALSE),
        shiny::numericInput("ncores", "Cores", .default_ncores(), min = 1, step = 1),
        shiny::actionButton("run", "Run simulation")
      ),
      shiny::mainPanel(
        shiny::h4("Results"),
        shiny::verbatimTextOutput("result"),
        shiny::p("For matrix-valued K or parameter grids, use sw_ni_power() or sw_ni_grid() in R.")
      )
    )
  )

  server <- function(input, output, session) {
    shiny::observeEvent(input$outcome, {
      shiny::updateNumericInput(
        session,
        "margin",
        value = if (identical(input$outcome, "binary")) 0.25 else 0.15
      )
    }, ignoreInit = TRUE)

    shiny::observeEvent(input$run, {
      output$result <- shiny::renderPrint({
        tryCatch({
          K <- .parse_numeric_text(input$K, "K", allow_blank = FALSE)
          clusters <- .parse_numeric_text(input$clusters_per_sequence, "clusters_per_sequence")
          period_effects <- .parse_numeric_text(input$period_effects, "period_effects")
          delay <- .parse_numeric_text(input$delay, "delay")

          shiny::withProgress(message = "Running simulations...", value = 0.5, {
            sw_ni_power(
              outcome = input$outcome,
              seed = input$seed,
              trials = input$trials,
              Tp = input$Tp,
              no_clusters_per_step = input$balanced_clusters,
              clusters_per_sequence = clusters,
              K = K,
              b1 = input$b1,
              mu = if (identical(input$outcome, "continuous")) input$mu else 5,
              sdw = if (identical(input$outcome, "continuous")) input$sdw else 2,
              sdc = input$sdc,
              theta = input$theta,
              margin = input$margin,
              delay = delay,
              period_effects = period_effects,
              adjust_period = input$adjust_period,
              sig.level = input$sig_level,
              REML = if (identical(input$outcome, "continuous")) input$REML else TRUE,
              failure_handling = input$failure_handling,
              parallel = input$parallel,
              ncores = input$ncores,
              return_details = TRUE
            )
          })
        }, error = function(e) {
          cat("Error:", conditionMessage(e), "\n")
        })
      })
    })
  }

  shiny::shinyApp(ui, server)
}
