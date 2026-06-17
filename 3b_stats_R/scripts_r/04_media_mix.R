# 04_media_mix.R — Ratio Photo / Vidéo / Texte par mois (global + par phase)
# Produit : 4_data_et_viz/04_media_mix_global.png, 05-07_media_mix_p{1,2,3}.png
# Rscript 3b_stats_R/scripts_r/04_media_mix.R

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
message("=== 04_media_mix.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_ratio_type_mois <- function(data, titre = "Répartition des types de posts",
                                 phase_lines = TRUE) {
  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      mois = as.Date(floor_date(date, "month")),
      type_pub = case_when(
        media_type == "photo" ~ "Photo",
        media_type == "video" ~ "Vidéo",
        TRUE ~ "Texte"
      )
    ) %>%
    count(mois, type_pub, name = "n") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      type_pub = c("Photo", "Texte", "Vidéo"),
      fill = list(n = 0)
    ) %>%
    mutate(type_pub = factor(type_pub, levels = c("Vidéo", "Photo", "Texte")))

  p <- ggplot(df, aes(x = mois, y = n, fill = type_pub)) +
    geom_col(position = "fill", color = "white", linewidth = 0.2, width = 25) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_cat() +
    labs(
      title    = titre,
      subtitle = "Part relative de chaque type (par mois)",
      x = NULL, y = "Proportion (%)",
      caption  = "Source : messages_clean.jsonl (1 365 messages). Texte = media_type manquant."
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_ratio_type_mois(df_clean, "Part Photo/Vidéo/Texte — Période complète"),
  file.path(OUT, "04_media_mix_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_type_mois(df_phase(1), "Part Photo/Vidéo/Texte — Phase 1", phase_lines = FALSE),
  file.path(OUT, "05_media_mix_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_type_mois(df_phase(2), "Part Photo/Vidéo/Texte — Phase 2", phase_lines = FALSE),
  file.path(OUT, "06_media_mix_p2.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_type_mois(df_phase(3), "Part Photo/Vidéo/Texte — Phase 3", phase_lines = FALSE),
  file.path(OUT, "07_media_mix_p3.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

message("=== Terminé : 4 exports dans ", OUT, " ===")
