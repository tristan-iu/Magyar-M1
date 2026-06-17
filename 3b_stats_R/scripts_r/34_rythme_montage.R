# 34_rythme_montage.R — Rythme de montage : durée moyenne de scène & cuts/min
# Produit : 4_data_et_viz/34a-34i (séries mensuelles, boxplots par phase, ECDF)
# Rscript 3b_stats_R/scripts_r/34_rythme_montage.R

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
message("=== 34_rythme_montage.R — ", nrow(df_clean), " messages chargés ===")

# S'assurer que les colonnes existent (ajoutées par scene_detect standalone)
df_clean <- df_clean |>
  ensure_col("scene_duree_moyenne", NA_real_) |>
  ensure_col("scene_coupes_par_min", NA_real_)

df_clean <- df_clean |>
  mutate(
    scene_duree_moyenne = suppressWarnings(as.numeric(scene_duree_moyenne)),
    scene_coupes_par_min = suppressWarnings(as.numeric(scene_coupes_par_min))
  )

# Corpus vidéo uniquement, avec données de montage
df_scene <- df_clean |>
  filter(
    media_type == "video",
    !is.na(mois),
    !is.na(scene_duree_moyenne) | !is.na(scene_coupes_par_min)
  ) |>
  mutate(mois = as.Date(mois))

n_videos <- nrow(df_scene)
message("  Vidéos avec données de montage : ", n_videos)

# Agrégation mensuelle — médiane robuste aux outliers
df_mois <- df_scene |>
  group_by(mois) |>
  summarise(
    scene_dur_med  = median(scene_duree_moyenne, na.rm = TRUE),
    cuts_min_med   = median(scene_coupes_par_min, na.rm = TRUE),
    n_videos       = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

# ---------------------------------------------------------------------------
# 34a — Durée moyenne de scène par mois (médiane)
# ---------------------------------------------------------------------------

plot_scene_dur <- function(df, titre = "Durée médiane de scène par mois") {
  ggplot(df, aes(x = mois, y = scene_dur_med)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["1_Artisanal"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " s", accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.10))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle — vidéos uniquement (taille = nombre de vidéos)",
      x = NULL, y = "Durée médiane de scène (s)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        n_videos, " vidéos avec données PySceneDetect."
      )
    )
}

# ---------------------------------------------------------------------------
# 34b — Cuts par minute par mois (médiane)
# ---------------------------------------------------------------------------

plot_cuts_min <- function(df, titre = "Cadence de montage (cuts/min) par mois") {
  ggplot(df, aes(x = mois, y = cuts_min_med)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["2_Semi-pro"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.10))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle — plus élevé = montage plus haché",
      x = NULL, y = "Cuts / minute (médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        n_videos, " vidéos avec données PySceneDetect."
      )
    )
}

# ---------------------------------------------------------------------------
# 34c — Overlay indexé : durée de scène vs cuts/min
# ---------------------------------------------------------------------------

plot_overlay_montage <- function(df,
  titre = "Rythme de montage : durée de scène & cadence (indexés)") {

  pal_ov <- c(
    "Durée médiane de scène"   = unname(PAL_PHASE["1_Artisanal"]),
    "Cuts / minute (médiane)"  = unname(PAL_PHASE["2_Semi-pro"])
  )

  # Inverser scene_dur pour que "haut = montage rapide" sur les deux axes
  df |>
    mutate(
      dur_idx  = to_index_100(scene_dur_med),
      cuts_idx = to_index_100(cuts_min_med)
    ) |>
    select(mois, dur_idx, cuts_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      dur_idx  = "Durée médiane de scène",
      cuts_idx = "Cuts / minute (médiane)"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    scale_colour_manual(values = pal_ov) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = titre,
      subtitle = "Indice min-max (0–100) calculé indépendamment pour chaque série",
      x = NULL, y = "Indice (0 = min, 100 = max)",
      caption  = "Source : messages_clean.jsonl. Montage accéléré = durée ↓ / cuts ↑."
    )
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_scene_dur(df_mois),
  file.path(OUT, "34a_scene_dur_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

save_plot(
  plot_cuts_min(df_mois),
  file.path(OUT, "34b_cuts_min_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

save_plot(
  plot_overlay_montage(df_mois),
  file.path(OUT, "34c_overlay_montage.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variantes avec changepoints
cpts_dur  <- compute_cpts(df_mois$scene_dur_med, df_mois$mois)
cpts_cuts <- compute_cpts(df_mois$cuts_min_med,  df_mois$mois)

if (length(cpts_dur) > 0) {
  p <- add_cpt_lines(
    plot_scene_dur(df_mois, "Durée de scène — ruptures détectées"),
    cpts_dur, color = PAL_PHASE["3_Institutionnel"]
  )
  save_plot(p, file.path(OUT, "34a_scene_dur_mois_cpt.png"),
            format = "wide_16_9", width = 10, dpi = 600)
}

if (length(cpts_cuts) > 0) {
  p <- add_cpt_lines(
    plot_cuts_min(df_mois, "Cuts/min — ruptures détectées"),
    cpts_cuts, color = PAL_PHASE["3_Institutionnel"]
  )
  save_plot(p, file.path(OUT, "34b_cuts_min_mois_cpt.png"),
            format = "wide_16_9", width = 10, dpi = 600)
}

# ---------------------------------------------------------------------------
# 34d — Boxplot cuts/min par phase (3 phases)
# ---------------------------------------------------------------------------

df_scene_ph <- df_scene |>
  mutate(phase_lbl = factor(case_when(
    phase == 1L ~ "1_Artisanal",
    phase == 2L ~ "2_Semi-pro",
    phase == 3L ~ "3_Institutionnel",
    TRUE ~ NA_character_
  ), levels = names(PAL_PHASE))) |>
  filter(!is.na(phase_lbl), !is.na(scene_coupes_par_min))

# Médianes par phase pour annotation
med_ph <- df_scene_ph |>
  group_by(phase_lbl) |>
  summarise(med = median(scene_coupes_par_min, na.rm = TRUE),
            n = n(), .groups = "drop")

plot_box_cuts_global <- function() {
  ggplot(df_scene_ph, aes(x = phase_lbl, y = scene_coupes_par_min, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_ph,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.1f", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    labs(
      title    = "Cadence de montage par phase",
      subtitle = paste0("Cuts/min — ", sum(med_ph$n), " vidéos avec données PySceneDetect"),
      x = NULL, y = "Cuts / minute",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(med_ph$phase_lbl, ": n=", med_ph$n, collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_cuts_global(),
  file.path(OUT, "34d_cuts_min_phase_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 34e — Boxplot cuts/min P1 vs P2 seulement
# ---------------------------------------------------------------------------

df_scene_p12 <- df_scene_ph |>
  filter(phase_lbl %in% c("1_Artisanal", "2_Semi-pro"))

med_p12 <- med_ph |>
  filter(phase_lbl %in% c("1_Artisanal", "2_Semi-pro"))

plot_box_cuts_p12 <- function() {
  ggplot(df_scene_p12, aes(x = phase_lbl, y = scene_coupes_par_min, fill = phase_lbl)) +
    geom_boxplot(width = 0.45, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_p12,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.1f", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    labs(
      title    = "Cadence de montage : Artisanal vs Semi-pro",
      subtitle = paste0("Cuts/min — P1 vs P2 uniquement (",
                        sum(med_p12$n), " vidéos)"),
      x = NULL, y = "Cuts / minute",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(med_p12$phase_lbl, ": n=", med_p12$n, collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_cuts_p12(),
  file.path(OUT, "34e_cuts_min_p12_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 34f — Boxplot durée moyenne de scène par phase (3 phases)
# ---------------------------------------------------------------------------

df_scene_dur_ph <- df_scene_ph |>
  filter(!is.na(scene_duree_moyenne))

med_dur_ph <- df_scene_dur_ph |>
  group_by(phase_lbl) |>
  summarise(med = median(scene_duree_moyenne, na.rm = TRUE),
            n = n(), .groups = "drop")

plot_box_dur_global <- function() {
  ggplot(df_scene_dur_ph, aes(x = phase_lbl, y = scene_duree_moyenne, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_dur_ph,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.1f s", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(suffix = " s", accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    coord_cartesian(ylim = c(0, 120)) +
    labs(
      title    = "Durée moyenne de scène par phase",
      subtitle = paste0("Médiane — ", sum(med_dur_ph$n),
                        " vidéos (zoom 0–120 s, outliers hors cadre)"),
      x = NULL, y = "Durée moyenne de scène (s)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(med_dur_ph$phase_lbl, ": n=", med_dur_ph$n, collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_dur_global(),
  file.path(OUT, "34f_scene_dur_phase_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 34g — Boxplot durée moyenne de scène P1 vs P2
# ---------------------------------------------------------------------------

df_scene_dur_p12 <- df_scene_dur_ph |>
  filter(phase_lbl %in% c("1_Artisanal", "2_Semi-pro"))

med_dur_p12 <- med_dur_ph |>
  filter(phase_lbl %in% c("1_Artisanal", "2_Semi-pro"))

plot_box_dur_p12 <- function() {
  ggplot(df_scene_dur_p12, aes(x = phase_lbl, y = scene_duree_moyenne, fill = phase_lbl)) +
    geom_boxplot(width = 0.45, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_dur_p12,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.1f s", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(suffix = " s", accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    coord_cartesian(ylim = c(0, 120)) +
    labs(
      title    = "Durée de scène : Artisanal vs Semi-pro",
      subtitle = paste0("Médiane — P1 vs P2 uniquement (",
                        sum(med_dur_p12$n),
                        " vidéos, zoom 0–120 s)"),
      x = NULL, y = "Durée moyenne de scène (s)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(med_dur_p12$phase_lbl, ": n=", med_dur_p12$n, collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_dur_p12(),
  file.path(OUT, "34g_scene_dur_p12_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 34h — ECDF cuts/min par phase : distribution cumulative (diversité)
# ---------------------------------------------------------------------------

plot_ecdf_cuts <- function() {
  # Statistiques textuelles pour annotations
  pct_zero <- df_scene_ph |>
    group_by(phase_lbl) |>
    summarise(pct0 = mean(scene_coupes_par_min == 0) * 100, .groups = "drop")

  ann_txt <- paste0(
    LBL_PHASE_SHORT[as.character(pct_zero$phase_lbl)],
    " : ", round(pct_zero$pct0, 0), "% à 0 cut"
  ) |> paste(collapse = "  |  ")

  ggplot(df_scene_ph, aes(x = scene_coupes_par_min, colour = phase_lbl)) +
    stat_ecdf(linewidth = 0.9, pad = FALSE) +
    geom_vline(xintercept = c(5, 10, 20),
               colour = "grey75", linetype = "dotted", linewidth = 0.4) +
    scale_colour_phase(short = TRUE) +
    scale_x_continuous(
      breaks = c(0, 5, 10, 15, 20, 30, 40, 50),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      labels = label_percent(),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    coord_cartesian(xlim = c(0, 50)) +
    labs(
      title    = "Distribution des cadences de montage par phase",
      subtitle = "Courbe cumulative (ECDF) — chaque point = % de vidéos ayant \u2264 X cuts/min",
      x = "Cuts / minute",
      y = "Proportion cumulée de vidéos",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        sum(med_ph$n), " vidéos. ", ann_txt, "."
      )
    ) +
    theme(legend.position = c(0.82, 0.35))
}

save_plot(
  plot_ecdf_cuts(),
  file.path(OUT, "34h_ecdf_cuts_min.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 34i — ECDF cuts/min P1 vs P2 seulement
# ---------------------------------------------------------------------------

plot_ecdf_cuts_p12 <- function() {
  pct_zero_12 <- df_scene_p12 |>
    group_by(phase_lbl) |>
    summarise(pct0 = mean(scene_coupes_par_min == 0) * 100, .groups = "drop")

  ann_txt <- paste0(
    LBL_PHASE_SHORT[as.character(pct_zero_12$phase_lbl)],
    " : ", round(pct_zero_12$pct0, 0), "% à 0 cut"
  ) |> paste(collapse = "  |  ")

  ggplot(df_scene_p12, aes(x = scene_coupes_par_min, colour = phase_lbl)) +
    stat_ecdf(linewidth = 0.9, pad = FALSE) +
    geom_vline(xintercept = c(5, 10, 20),
               colour = "grey75", linetype = "dotted", linewidth = 0.4) +
    scale_colour_phase(short = TRUE) +
    scale_x_continuous(
      breaks = c(0, 5, 10, 15, 20, 30, 40, 50),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      labels = label_percent(),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    coord_cartesian(xlim = c(0, 50)) +
    labs(
      title    = "Distribution des cadences : Artisanal vs Semi-pro",
      subtitle = "ECDF — P1 vs P2 uniquement",
      x = "Cuts / minute",
      y = "Proportion cumulée de vidéos",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        sum(med_p12$n), " vidéos. ", ann_txt, "."
      )
    ) +
    theme(legend.position = c(0.82, 0.35))
}

save_plot(
  plot_ecdf_cuts_p12(),
  file.path(OUT, "34i_ecdf_cuts_min_p12.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Résumé stats par phase
df_phase_stats <- df_scene |>
  mutate(phase_lbl = case_when(
    phase == 1L ~ "1_Artisanal",
    phase == 2L ~ "2_Semi-pro",
    phase == 3L ~ "3_Institutionnel",
    TRUE ~ NA_character_
  )) |>
  filter(!is.na(phase_lbl)) |>
  group_by(phase_lbl) |>
  summarise(
    n            = n(),
    dur_med      = round(median(scene_duree_moyenne, na.rm = TRUE), 2),
    dur_mean     = round(mean(scene_duree_moyenne, na.rm = TRUE), 2),
    cuts_min_med = round(median(scene_coupes_par_min, na.rm = TRUE), 2),
    cuts_min_mean= round(mean(scene_coupes_par_min, na.rm = TRUE), 2),
    .groups = "drop"
  )

message("\n=== Rythme de montage par phase ===")
print(as.data.frame(df_phase_stats))

message("=== Terminé : exports dans ", OUT, " ===")
