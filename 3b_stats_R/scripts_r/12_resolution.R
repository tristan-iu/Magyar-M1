# 12_resolution.R — Répartition et évolution des résolutions (global + phases)
# Produit : 4_data_et_viz/12_resolution_global.png, 13-16_resolution_mois_*.png
# Rscript 3b_stats_R/scripts_r/12_resolution.R

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
message("=== 12_resolution.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------

plot_ratio_resolution <- function(data, top_n = 12,
                                  titre = "Répartition des résolutions (période complète)") {
  df <- data %>%
    filter(media_type %in% c("photo", "video")) %>%
    filter(!is.na(largeur), !is.na(hauteur)) %>%
    mutate(resolution = paste0(largeur, "x", hauteur)) %>%
    count(resolution, name = "n") %>%
    arrange(desc(n)) %>%
    mutate(resolution = if_else(row_number() <= top_n, resolution, "Autres")) %>%
    group_by(resolution) %>%
    summarise(n = sum(n), .groups = "drop") %>%
    mutate(prop = n / sum(n)) %>%
    arrange(prop)

  ggplot(df, aes(x = resolution, y = prop)) +
    geom_col(width = 0.8, fill = unname(PAL_PHASE["1_Artisanal"])) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = titre,
      subtitle = paste0("Top ", top_n, " + Autres (photo + vidéo)"),
      x = NULL, y = "Proportion (%)",
      caption  = "Vidéos et photos, dimensions connues."
    )
}

plot_ratio_resolution_mois <- function(data, top_k = 4,
                                       titre = "Évolution des résolutions (mensuel)",
                                       phase_lines = TRUE) {
  base <- data %>%
    filter(!is.na(date)) %>%
    mutate(mois = as.Date(floor_date(date, "month"))) %>%
    filter(media_type %in% c("photo", "video")) %>%
    filter(!is.na(largeur), !is.na(hauteur)) %>%
    mutate(resolution = paste0(largeur, "x", hauteur))

  top_res <- base %>%
    count(resolution, sort = TRUE, name = "n") %>%
    slice_head(n = top_k) %>%
    pull(resolution)

  df <- base %>%
    mutate(resolution_grp = if_else(resolution %in% top_res, resolution, "Autres")) %>%
    count(mois, resolution_grp, name = "n") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      resolution_grp = c(sort(top_res), "Autres"),
      fill = list(n = 0)
    ) %>%
    mutate(resolution_grp = factor(resolution_grp, levels = c(sort(top_res), "Autres")))

  check_n_categories(df$resolution_grp, var_name = "résolution")

  p <- ggplot(df, aes(x = mois, y = n, fill = resolution_grp)) +
    geom_col(position = "fill", color = "white", linewidth = 0.2, width = 25) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_cat() +
    labs(
      title    = titre,
      subtitle = paste0("Top ", top_k, " + Autres (photo + vidéo)"),
      x = NULL, y = "Proportion (%)",
      caption  = "Vidéos et photos, dimensions connues."
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_ratio_resolution(df_clean, top_n = 12),
  file.path(OUT, "12_resolution_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_resolution_mois(df_clean, top_k = 4, titre = "Résolutions — période complète"),
  file.path(OUT, "13_resolution_mois_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_resolution_mois(df_phase(1), top_k = 4, titre = "Résolutions — phase 1", phase_lines = FALSE),
  file.path(OUT, "14_resolution_mois_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_resolution_mois(df_phase(2), top_k = 4, titre = "Résolutions — phase 2", phase_lines = FALSE),
  file.path(OUT, "15_resolution_mois_p2.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_ratio_resolution_mois(df_phase(3), top_k = 4, titre = "Résolutions — phase 3", phase_lines = FALSE),
  file.path(OUT, "16_resolution_mois_p3.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

message("=== Terminé : 5 exports dans ", OUT, " ===")
