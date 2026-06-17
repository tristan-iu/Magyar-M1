# 08c_orientation_p12.R — Ratio orientation mensuel P1+P2, avec et sans changepoint
# Produit : 4_data_et_viz/08c_orientation_p12{,_cpt}.png
# Rscript 3b_stats_R/scripts_r/08c_orientation_p12.R

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
message("=== 08c_orientation_p12.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Données P1 + P2
# ---------------------------------------------------------------------------

df_p12_ori <- df_phase(c(1, 2)) %>%
  filter(
    !is.na(date),
    media_type %in% c("video", "photo"),
    !is.na(orientation),
    orientation %in% c("vertical", "horizontal", "square")
  ) %>%
  mutate(
    mois = as.Date(floor_date(date, "month")),
    orientation = recode(orientation,
      "vertical"   = "Vertical",
      "horizontal" = "Horizontal",
      "square"     = "Carré"
    )
  ) %>%
  count(mois, orientation, name = "n") %>%
  complete(
    mois = seq(min(mois), max(mois), by = "month"),
    orientation = c("Vertical", "Carré", "Horizontal"),
    fill = list(n = 0)
  ) %>%
  mutate(orientation = factor(orientation, levels = c("Vertical", "Carré", "Horizontal")))

# ---------------------------------------------------------------------------
# Plot de base (réutilisé pour les deux versions)
# ---------------------------------------------------------------------------

make_plot <- function(data) {
  ggplot(data, aes(x = mois, y = n, fill = orientation)) +
    geom_col(position = "fill", color = "white", linewidth = 0.2, width = 25) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_cat() +
    # Borne P1/P2 sourcée depuis r_source.R (source unique = config.yaml)
    geom_phase_lines(
      dates  = bornes$p2[1],
      labels = "P1\u2192P2"
    ) +
    labs(
      title    = "Évolution des formats vidéo/photo (sept. 2022 – sept. 2024)",
      subtitle = NULL,
      x = NULL, y = "Proportion (%)",
      caption  = NULL
    )
}

# ---------------------------------------------------------------------------
# 1) Version sans changepoint
# ---------------------------------------------------------------------------

p_base <- make_plot(df_p12_ori)

save_plot(p_base, file.path(OUT, "08c_orientation_p12.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 2) Version avec changepoint sur la série % horizontal
# ---------------------------------------------------------------------------

# Série mensuelle du % horizontal
df_horiz <- df_p12_ori %>%
  group_by(mois) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  filter(orientation == "Horizontal")

cpt_dates <- compute_cpts(df_horiz$pct, df_horiz$mois)
message("Changepoints détectés : ", paste(cpt_dates, collapse = ", "))

p_cpt <- make_plot(df_p12_ori)
p_cpt <- add_cpt_lines(p_cpt, cpt_dates)
p_cpt <- p_cpt +
  labs(subtitle = "Ratio mensuel — avec ruptures (changepoint sur % Horizontal)")

save_plot(p_cpt, file.path(OUT, "08c_orientation_p12_cpt.png"),
          format = "wide_16_9", width = 10, dpi = 600)

message("=== Terminé : 2 exports dans ", OUT, " ===")
