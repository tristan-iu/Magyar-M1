# 08b_orientation_h.R — Ratio orientation H/V jusqu'à oct. 2024, barres horizontales
# Produit : 4_data_et_viz/08b_orientation_ratio_h.png
# Rscript 3b_stats_R/scripts_r/08b_orientation_h.R

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
message("=== 08b_orientation_h.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Données : filtrer jusqu'à oct. 2024 inclus, médias avec orientation connue
# ---------------------------------------------------------------------------

df_ori <- df_clean %>%
  filter(
    !is.na(date),
    jour <= as.Date("2024-10-31"),
    media_type %in% c("video", "photo"),
    !is.na(orientation),
    orientation %in% c("vertical", "horizontal", "square")
  ) %>%
  mutate(
    orientation = recode(orientation,
      "vertical"   = "Vertical",
      "horizontal" = "Horizontal",
      "square"     = "Carré"
    ),
    phase_lbl = case_when(
      phase == 1L ~ "P1 — Artisanal\n(sept. 2022 – déc. 2023)",
      phase == 2L ~ "P2 — Semi-pro\n(janv. – sept. 2024)",
      TRUE        ~ "Oct. 2024"
    ),
    phase_lbl = factor(phase_lbl, levels = c(
      "P1 — Artisanal\n(sept. 2022 – déc. 2023)",
      "P2 — Semi-pro\n(janv. – sept. 2024)",
      "Oct. 2024"
    ))
  )

# Compter et calculer les proportions
df_counts <- df_ori %>%
  count(phase_lbl, orientation, name = "n") %>%
  group_by(phase_lbl) %>%
  mutate(
    total = sum(n),
    pct   = n / total
  ) %>%
  ungroup() %>%
  mutate(orientation = factor(orientation, levels = c("Horizontal", "Carré", "Vertical")))

# ---------------------------------------------------------------------------
# Plot — barres horizontales empilées (100 %)
# ---------------------------------------------------------------------------

p <- ggplot(df_counts, aes(x = phase_lbl, y = pct, fill = orientation)) +
  geom_col(position = "fill", width = 0.65, color = "white", linewidth = 0.3) +
  geom_text(
    aes(label = ifelse(pct > 0.04, sprintf("%d\n(%.0f%%)", n, pct * 100), "")),
    position = position_fill(vjust = 0.5),
    size = 3.5, fontface = "bold", color = "white"
  ) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), expand = c(0, 0)) +
  scale_fill_cat() +
  labs(
    title    = "Ratio des formats vidéo/photo — jusqu'à oct. 2024",
    subtitle = "Proportion Horizontal / Carré / Vertical par phase",
    x = NULL, y = "Proportion",
    caption  = "Source : messages_clean.jsonl. Filtre : media_type ∈ {photo, video}, date ≤ oct. 2024."
  ) +
  theme_madyar() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3)
  )

save_plot(p, file.path(OUT, "08b_orientation_ratio_h.png"),
          format = "wide_16_9", width = 10, dpi = 600)

message("=== Terminé ===")
