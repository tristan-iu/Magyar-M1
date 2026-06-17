# 08_orientation.R — Ratio orientation Vertical / Horizontal / Carré (global + phases)
# Produit : 4_data_et_viz/08_orientation_global.png, 09-11_orientation_p{1,2,3}.png
# Rscript 3b_stats_R/scripts_r/08_orientation.R

this_file <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  if (length(f) > 0 && nzchar(f)) return(normalizePath(f, mustWork = FALSE))
  for (env in rev(sys.frames()))
    if (!is.null(env$ofile)) return(normalizePath(env$ofile, mustWork = FALSE))
  stop("Lancer via Rscript ou source().")
})
BASE <- dirname(dirname(this_file))
source(file.path(BASE, "r_source.R"))
OUT  <- file.path(BASE, "..", "4_data_et_viz")
dir.create(OUT, showWarnings = FALSE)
message("=== 08_orientation.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_ratio_orientation_mois <- function(data, titre = "Évolution des formats vidéo/photo",
                                        phase_lines = TRUE) {
  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      mois = as.Date(floor_date(date, "month")),
      orientation = tolower(as.character(orientation)),
      orientation = recode(orientation,
        "vertical"   = "Vertical",
        "horizontal" = "Horizontal",
        "square"     = "Carré",
        .default = NA_character_
      )
    ) %>%
    filter(media_type %in% c("video", "photo"), !is.na(orientation)) %>%
    count(mois, orientation, name = "n") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      orientation = c("Vertical", "Carré", "Horizontal"),
      fill = list(n = 0)
    ) %>%
    mutate(orientation = factor(orientation, levels = c("Vertical", "Carré", "Horizontal")))

  p <- ggplot(df, aes(x = mois, y = n, fill = orientation)) +
    geom_col(position = "fill", color = "white", linewidth = 0.2, width = 25) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_cat() +
    labs(
      title    = titre,
      subtitle = "Ratio de publications par format (vidéos + photos)",
      x = NULL, y = "Proportion (%)",
      caption  = "Source : messages_clean.jsonl. Filtre : media_type ∈ {photo, video}."
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_ratio_orientation_mois(df_clean, "Évolution des formats — Période complète"),
  file.path(OUT, "08_orientation_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_orientation_mois(df_phase(1), "Évolution des formats — Phase 1", phase_lines = FALSE),
  file.path(OUT, "09_orientation_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_orientation_mois(df_phase(2), "Évolution des formats — Phase 2", phase_lines = FALSE),
  file.path(OUT, "10_orientation_p2.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_orientation_mois(df_phase(3), "Évolution des formats — Phase 3", phase_lines = FALSE),
  file.path(OUT, "11_orientation_p3.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

message("=== Terminé : 4 exports dans ", OUT, " ===")
