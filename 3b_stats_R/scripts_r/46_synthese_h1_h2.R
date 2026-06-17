# 46_synthese_h1_h2.R — Figure de synthèse : overlay indexé 0–100 des métriques mensuelles
# Produit : 4_data_et_viz/46a_synthese_h1.png, 46b_synthese_h2.png,
#           46c_synthese_globale.png, 46e_heatmap_synthese.png
# Rscript 3b_stats_R/scripts_r/46_synthese_h1_h2.R

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
message("=== 46_synthese_h1_h2.R — ", nrow(df_clean), " messages chargés ===")

# Colonnes nécessaires
df_clean <- df_clean |>
  ensure_col("scene_coupes_par_min", NA_real_) |>
  ensure_col("visages_magyar_ratio", NA_real_) |>
  ensure_col("video_bitrate", NA_real_) |>
  ensure_col("fichier_taille", NA_real_)

# Vidéos seulement (pour les métriques techniques)
df_vid <- df_clean |>
  filter(media_type == "video", !is.na(mois)) |>
  mutate(
    mois = as.Date(mois),
    scene_coupes_par_min = suppressWarnings(as.numeric(scene_coupes_par_min)),
    visages_magyar_ratio = suppressWarnings(as.numeric(visages_magyar_ratio)),
    video_bitrate = suppressWarnings(as.numeric(video_bitrate)),
    file_size_mb = suppressWarnings(as.numeric(fichier_taille)) / (1024^2)
  )

# ---------------------------------------------------------------------------
# Agrégation mensuelle de toutes les métriques
# ---------------------------------------------------------------------------

df_synth <- df_vid |>
  group_by(mois) |>
  summarise(
    duree_med       = median(duree_sec, na.rm = TRUE),
    cuts_min_med    = median(scene_coupes_par_min, na.rm = TRUE),
    resolution_med  = median(largeur * hauteur, na.rm = TRUE),
    filesize_med    = median(file_size_mb, na.rm = TRUE),
    speech_ratio_med = median(parole_ratio, na.rm = TRUE),
    faces_magyar_med = median(visages_magyar_ratio, na.rm = TRUE),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month")) |>
  arrange(mois)

# Indexation 0–100 (chaque série indépendamment)
df_idx <- df_synth |>
  mutate(
    # durée inversée : bas = vidéos courtes
    duree_idx      = 100 - to_index_100(duree_med),
    cuts_idx       = to_index_100(cuts_min_med),
    resolution_idx = to_index_100(resolution_med),
    filesize_idx   = to_index_100(filesize_med),
    speech_idx     = to_index_100(speech_ratio_med),
    faces_idx      = to_index_100(faces_magyar_med)
  )

# ---------------------------------------------------------------------------
# 46a — Synthèse (forme)
# ---------------------------------------------------------------------------

pal_h1 <- c(
  "Durée (inversé : court = haut)" = "#2166AC",
  "Cuts/min"                        = "#D95F02",
  "Résolution"                      = "#1B9E77",
  "Taille fichier"                  = "#7570B3"
)

plot_h1 <- function(df) {
  df |>
    select(mois, duree_idx, cuts_idx, resolution_idx, filesize_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      duree_idx      = "Durée (inversé : court = haut)",
      cuts_idx       = "Cuts/min",
      resolution_idx = "Résolution",
      filesize_idx   = "Taille fichier"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 1.5, na.rm = TRUE) +
    scale_colour_manual(values = pal_h1) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "H1 — Industrialisation de la forme",
      subtitle = "Indices 0–100 (médiane mensuelle) — convergence vers le haut = professionnalisation",
      x = NULL, y = "Indice (0 = min corpus, 100 = max corpus)",
      caption  = "Source : messages_clean.jsonl. Durée inversée (100 = vidéo courte). Chaque série normalisée indépendamment."
    )
}

save_plot(
  plot_h1(df_idx),
  file.path(OUT, "46a_synthese_h1.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# ---------------------------------------------------------------------------
# 46b — Synthèse (dépersonnalisation)
# ---------------------------------------------------------------------------

pal_h2 <- c(
  "Speech ratio (voix)"         = "#D95F02",
  "Présence Magyar (visage)"    = "#2166AC"
)

plot_h2 <- function(df) {
  df |>
    select(mois, speech_idx, faces_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      speech_idx = "Speech ratio (voix)",
      faces_idx  = "Présence Magyar (visage)"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(linewidth = 0.9, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    scale_colour_manual(values = pal_h2) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "H2 — Dépersonnalisation du discours",
      subtitle = "Indices 0–100 — convergence vers 0 = disparition de Magyar",
      x = NULL, y = "Indice (100 = maximum présence, 0 = absence)",
      caption  = "Source : messages_clean.jsonl. Speech ratio (Whisper/Silero) + InsightFace."
    )
}

save_plot(
  plot_h2(df_idx),
  file.path(OUT, "46b_synthese_h2.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# ---------------------------------------------------------------------------
# 46c — SYNTHÈSE GLOBALE : 6 séries indexées sur un seul graphique
# ---------------------------------------------------------------------------

pal_all <- c(
  "Durée (inv.)"      = "#2166AC",
  "Cuts/min"           = "#1B9E77",
  "Taille fichier"     = "#7570B3",
  "Visage Magyar"      = "#D95F02",
  "Voix (speech)"      = "#E7298A",
  "Résolution"         = "#66A61E"
)

plot_synthese <- function(df) {
  df |>
    select(mois, duree_idx, cuts_idx, resolution_idx,
           speech_idx, faces_idx, filesize_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      duree_idx      = "Durée (inv.)",
      cuts_idx       = "Cuts/min",
      resolution_idx = "Résolution",
      filesize_idx   = "Taille fichier",
      speech_idx     = "Voix (speech)",
      faces_idx      = "Visage Magyar"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 1.3, na.rm = TRUE) +
    scale_colour_manual(values = pal_all) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "Synthèse : trajectoire de professionnalisation",
      subtitle = "6 indicateurs indexés (0–100) — H1 monte, H2 descend",
      x = NULL, y = "Indice normalisé (0–100)",
      caption  = paste0(
        "Source : messages_clean.jsonl. Médiane mensuelle. ",
        "Durée inversée (haut = court). Chaque série normalisée indépendamment."
      )
    )
}

save_plot(
  plot_synthese(df_idx),
  file.path(OUT, "46c_synthese_globale.png"),
  format = "wide_16_9", width = 14, dpi = 600
)

# ---------------------------------------------------------------------------
# 46d — Tableau récapitulatif par phase (médiane des métriques brutes)
# ---------------------------------------------------------------------------

df_recap <- df_vid |>
  mutate(phase_lbl = factor(case_when(
    phase == 1L ~ "1_Artisanal",
    phase == 2L ~ "2_Semi-pro",
    phase == 3L ~ "3_Institutionnel",
    TRUE ~ NA_character_
  ), levels = names(PAL_PHASE))) |>
  filter(!is.na(phase_lbl)) |>
  group_by(phase_lbl) |>
  summarise(
    n = n(),
    duree_med_s     = round(median(duree_sec, na.rm = TRUE), 0),
    cuts_min_med    = round(median(scene_coupes_par_min, na.rm = TRUE), 1),
    width_med       = round(median(largeur, na.rm = TRUE), 0),
    height_med      = round(median(hauteur, na.rm = TRUE), 0),
    filesize_med_mb = round(median(file_size_mb, na.rm = TRUE), 1),
    speech_ratio_med = round(median(parole_ratio, na.rm = TRUE), 3),
    faces_magyar_med = round(median(visages_magyar_ratio, na.rm = TRUE), 3),
    .groups = "drop"
  )

message("\n=== Récapitulatif par phase ===")
print(as.data.frame(df_recap))

# ---------------------------------------------------------------------------
# 46e — Heatmap résumé : métrique × phase (tile plot)
# ---------------------------------------------------------------------------

# Normaliser chaque métrique 0-1 pour le heatmap
df_tile <- df_recap |>
  select(phase_lbl, duree_med_s, cuts_min_med, filesize_med_mb,
         speech_ratio_med, faces_magyar_med) |>
  pivot_longer(-phase_lbl, names_to = "metrique", values_to = "val") |>
  group_by(metrique) |>
  mutate(val_norm = (val - min(val)) / max(max(val) - min(val), 1e-9)) |>
  ungroup() |>
  mutate(metrique = recode(metrique,
    duree_med_s      = "Durée (s)",
    cuts_min_med     = "Cuts/min",
    filesize_med_mb  = "Taille (Mo)",
    speech_ratio_med = "Speech ratio",
    faces_magyar_med = "Visage Magyar"
  ))

plot_heatmap_phase <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = metrique, fill = val_norm)) +
    geom_tile(colour = "white", linewidth = 1.5) +
    geom_text(aes(label = round(val, 1)), size = 4, fontface = "bold") +
    scale_fill_gradient(low = PAL_SEQ_LOW, high = PAL_SEQ_HIGH,
                        name = "Normalisé\n(0–1)") +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    labs(
      title    = "Heatmap de synthèse : métriques × phases",
      subtitle = "Valeurs brutes affichées — gradient = position relative",
      x = NULL, y = NULL,
      caption  = "Source : messages_clean.jsonl. Médiane par phase."
    ) +
    theme(
      axis.text.y = element_text(size = 12),
      panel.grid = element_blank()
    )
}

save_plot(
  plot_heatmap_phase(df_tile),
  file.path(OUT, "46e_heatmap_synthese.png"),
  format = "square", width = 10, dpi = 600
)

message("=== Terminé : exports dans ", OUT, " ===")
