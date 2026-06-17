# 43_grouped_messages.R — Albums (messages groupés)
# album_id non-null = post faisant partie d'un album multi-média
# Produit : 4_data_et_viz/43a_pct_album_mois.png, 43b_album_size_boxplot.png, 43c_album_mix_phase.png
# Rscript 3b_stats_R/scripts_r/43_grouped_messages.R

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
message("=== 43_grouped_messages.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Préparation
# ---------------------------------------------------------------------------

df_grp <- df_clean |>
  filter(!is.na(mois), has_media) |>
  mutate(
    mois = as.Date(mois),
    is_album = !is.na(album_id) & album_id != "" & album_id != "NA",
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  )

n_media <- nrow(df_grp)
n_albums_msg <- sum(df_grp$is_album)
message("  Médias : ", n_media, " dont ", n_albums_msg, " en album")

# Taille des albums (nombre de médias par album_id)
df_album_size <- df_grp |>
  filter(is_album) |>
  group_by(album_id) |>
  summarise(
    album_size = n(),
    mois = min(mois),
    jour = min(jour),
    phase = first(phase),
    phase_lbl = first(phase_lbl),
    .groups = "drop"
  )

n_albums <- nrow(df_album_size)
message("  Albums distincts : ", n_albums, " (taille médiane : ",
        median(df_album_size$album_size), ")")

# ---------------------------------------------------------------------------
# 43a — % de médias en album par mois
# ---------------------------------------------------------------------------

df_mois_album <- df_grp |>
  group_by(mois) |>
  summarise(
    pct_album = mean(is_album) * 100,
    n_media   = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

plot_pct_album <- function(df, titre = "Part des médias en album par mois") {
  ggplot(df, aes(x = mois, y = pct_album)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_col(fill = PAL_PHASE["1_Artisanal"], width = 25, alpha = 0.85, na.rm = TRUE) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " %", accuracy = 1),
      limits = c(0, 100),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title    = titre,
      subtitle = "% mensuel de médias faisant partie d'un album (album_id non-null)",
      x = NULL, y = "Médias en album (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_media,
        " médias, ", n_albums, " albums distincts."
      )
    )
}

save_plot(
  plot_pct_album(df_mois_album),
  file.path(OUT, "43a_pct_album_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 43b — Taille des albums par phase (boxplot)
# ---------------------------------------------------------------------------

df_album_ph <- df_album_size |> filter(!is.na(phase_lbl))

med_album <- df_album_ph |>
  group_by(phase_lbl) |>
  summarise(med = median(album_size), n = n(), .groups = "drop")

plot_album_size <- function() {
  ggplot(df_album_ph, aes(x = phase_lbl, y = album_size, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_album,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %d", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      breaks = seq(0, 20, 2),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    labs(
      title    = "Taille des albums par phase",
      subtitle = paste0("Nombre de médias par album — ", n_albums, " albums"),
      x = NULL, y = "Médias par album",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d albums", LBL_PHASE_SHORT[as.character(med_album$phase_lbl)],
              med_album$n), collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_album_size(),
  file.path(OUT, "43b_album_size_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 43c — Composition des albums : mix vidéo/photo par phase
# ---------------------------------------------------------------------------

df_album_mix <- df_grp |>
  filter(is_album, !is.na(phase_lbl)) |>
  mutate(type_simple = case_when(
    media_type == "video" ~ "Vidéo",
    media_type == "photo" ~ "Photo",
    TRUE ~ "Autre"
  )) |>
  count(phase_lbl, type_simple) |>
  group_by(phase_lbl) |>
  mutate(pct = n / sum(n)) |>
  ungroup()

plot_album_mix <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = pct, fill = type_simple)) +
    geom_col(position = "fill", colour = "white", linewidth = 0.3) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_fill_cat() +
    labs(
      title    = "Composition des albums par phase",
      subtitle = "Part vidéo vs photo dans les posts groupés",
      x = NULL, y = "Proportion (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ",
        n_albums_msg, " médias en album."
      )
    )
}

save_plot(
  plot_album_mix(df_album_mix),
  file.path(OUT, "43c_album_mix_phase.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Résumé
message("\n=== Albums par phase ===")
print(as.data.frame(med_album))
message("\n=== % albums par phase ===")
df_grp |>
  filter(!is.na(phase_lbl)) |>
  group_by(phase_lbl) |>
  summarise(pct = round(mean(is_album) * 100, 1), n = n(), .groups = "drop") |>
  as.data.frame() |>
  print()
message("=== Terminé : exports dans ", OUT, " ===")
