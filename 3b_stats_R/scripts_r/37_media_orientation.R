# 37_media_orientation.R — Répartition photo/vidéo × orientation par mois
# Produit : 4_data_et_viz/37a_media_orient_global.png, 37b_media_orient_count_global.png
# Rscript 3b_stats_R/scripts_r/37_media_orientation.R

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
message("=== 37_media_orientation.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# PRÉPARATION
# ---------------------------------------------------------------------------

df_mo <- df_clean %>%
  filter(!is.na(date), media_type %in% c("video", "photo")) %>%
  mutate(
    mois = as.Date(floor_date(date, "month")),
    type_media = factor(
      ifelse(media_type == "video", "Vidéo", "Photo"),
      levels = c("Vidéo", "Photo")
    ),
    orient = tolower(as.character(orientation)),
    orient = recode(orient,
      "vertical"   = "Vertical",
      "horizontal" = "Horizontal",
      "square"     = "Carré",
      .default     = NA_character_
    ),
    orient = factor(orient, levels = c("Vertical", "Horizontal", "Carré"))
  ) %>%
  filter(!is.na(orient))

# Catégorie croisée type × orientation
df_mo <- df_mo %>%
  mutate(
    cat = paste(type_media, orient, sep = " — "),
    cat = factor(cat, levels = c(
      "Vidéo — Vertical", "Vidéo — Horizontal", "Vidéo — Carré",
      "Photo — Vertical", "Photo — Horizontal", "Photo — Carré"
    ))
  )

# Palette : 6 catégories, PAL_CAT (Dark2)
pal_mo <- setNames(PAL_CAT[1:6], levels(df_mo$cat))

# ---------------------------------------------------------------------------
# PLOT — stacked bar (proportion) par mois
# ---------------------------------------------------------------------------

plot_media_orient <- function(data, titre, phase_lines = TRUE,
                              date_breaks = "3 months") {
  df <- data %>%
    count(mois, cat, name = "n") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      cat  = levels(df_mo$cat),
      fill = list(n = 0)
    )

  p <- ggplot(df, aes(x = mois, y = n, fill = cat)) +
    geom_col(position = "fill", colour = "white", linewidth = 0.2, width = 25) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = date_breaks) +
    scale_fill_manual(values = pal_mo) +
    labs(
      title    = titre,
      subtitle = "Part relative mensuelle — type de média × orientation",
      x = NULL, y = "Proportion (%)",
      caption  = paste0("Source : messages_clean.jsonl. Filtre : photo + vidéo avec orientation connue (",
                        nrow(data), " médias).")
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# PLOT — counts absolus empilés par mois
# ---------------------------------------------------------------------------

plot_media_orient_count <- function(data, titre, phase_lines = TRUE,
                                    date_breaks = "3 months") {
  df <- data %>%
    count(mois, cat, name = "n") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      cat  = levels(df_mo$cat),
      fill = list(n = 0)
    )

  p <- ggplot(df, aes(x = mois, y = n, fill = cat)) +
    geom_col(colour = "white", linewidth = 0.2, width = 25) +
    scale_x_mois(breaks = date_breaks) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    scale_fill_manual(values = pal_mo) +
    labs(
      title    = titre,
      subtitle = "Nombre de publications par mois — type de média × orientation",
      x = NULL, y = "Nombre de publications",
      caption  = paste0("Source : messages_clean.jsonl. Photo + vidéo avec orientation connue (",
                        nrow(data), " médias).")
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

# Global — proportions
save_plot(
  plot_media_orient(df_mo, "Répartition photo/vidéo × orientation — Période complète"),
  file.path(OUT, "37a_media_orient_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Global — counts
save_plot(
  plot_media_orient_count(df_mo, "Volume photo/vidéo × orientation — Période complète"),
  file.path(OUT, "37b_media_orient_count_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Par phase — proportions
for (ph in 1:3) {
  lbl <- c("Phase 1 — Artisanal", "Phase 2 — Semi-pro", "Phase 3 — Institutionnel")[ph]
  sub <- df_mo %>% filter(phase == ph)
  if (nrow(sub) == 0) next
  save_plot(
    plot_media_orient(sub, paste("Répartition média × orientation —", lbl),
                      phase_lines = FALSE, date_breaks = "2 months"),
    file.path(OUT, sprintf("37c_media_orient_P%d.png", ph)),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

message("=== Terminé : 5 exports dans ", OUT, " ===")
