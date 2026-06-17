# 39_speech_ratio.R — Speech ratio (parole_ratio) par mois et par phase
# Produit : 4_data_et_viz/39a_speech_ratio_mois{,_cpt}.png, 39b_pct_parole_mois.png,
#           39c_speech_ratio_phase_boxplot.png, 39d_overlay_voix_visage.png
# Rscript 3b_stats_R/scripts_r/39_speech_ratio.R

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
message("=== 39_speech_ratio.R — ", nrow(df_clean), " messages chargés ===")

# Corpus vidéo uniquement (parole_ratio n'a de sens que pour les vidéos)
df_vid <- df_clean |>
  filter(media_type == "video", !is.na(mois)) |>
  mutate(
    mois = as.Date(mois),
    parole_ratio = suppressWarnings(as.numeric(parole_ratio)),
    has_speech_flag = !is.na(parole_ratio) & parole_ratio > 0
  )

n_videos <- nrow(df_vid)
message("  Vidéos : ", n_videos)

# ---------------------------------------------------------------------------
# 39a — Speech ratio médian par mois (série temporelle)
# ---------------------------------------------------------------------------

df_mois_sr <- df_vid |>
  group_by(mois) |>
  summarise(
    sr_median  = median(parole_ratio, na.rm = TRUE),
    sr_mean    = mean(parole_ratio, na.rm = TRUE),
    pct_parole = mean(has_speech_flag, na.rm = TRUE) * 100,
    n_videos   = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

plot_sr_mois <- function(df, titre = "Ratio de parole par mois") {
  ggplot(df, aes(x = mois, y = sr_median)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["2_Semi-pro"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_percent(accuracy = 1),
      limits = c(0, NA),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle du parole_ratio (parole / durée totale)",
      x = NULL, y = "Speech ratio (médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_videos,
        " vidéos. Whisper large-v3 + Silero VAD."
      )
    )
}

save_plot(
  plot_sr_mois(df_mois_sr),
  file.path(OUT, "39a_speech_ratio_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variante avec changepoints
cpts_sr <- compute_cpts(df_mois_sr$sr_median, df_mois_sr$mois)
if (length(cpts_sr) > 0) {
  save_plot(
    add_cpt_lines(
      plot_sr_mois(df_mois_sr, "Speech ratio — ruptures détectées"),
      cpts_sr, color = PAL_PHASE["3_Institutionnel"]
    ),
    file.path(OUT, "39a_speech_ratio_mois_cpt.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# ---------------------------------------------------------------------------
# 39b — % de vidéos avec parole par mois
# ---------------------------------------------------------------------------

plot_pct_parole <- function(df, titre = "Part des vidéos avec parole détectée") {
  ggplot(df, aes(x = mois, y = pct_parole)) +
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
      subtitle = "Pourcentage mensuel de vidéos où parole_present = TRUE",
      x = NULL, y = "Vidéos avec parole (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_videos,
        " vidéos. Silero VAD (parole_ratio > 0)."
      )
    )
}

save_plot(
  plot_pct_parole(df_mois_sr),
  file.path(OUT, "39b_pct_parole_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 39c — Boxplot parole_ratio par phase
# ---------------------------------------------------------------------------

df_vid_ph <- df_vid |>
  mutate(phase_lbl = factor(case_when(
    phase == 1L ~ "1_Artisanal",
    phase == 2L ~ "2_Semi-pro",
    phase == 3L ~ "3_Institutionnel",
    TRUE ~ NA_character_
  ), levels = names(PAL_PHASE))) |>
  filter(!is.na(phase_lbl))

med_sr <- df_vid_ph |>
  group_by(phase_lbl) |>
  summarise(
    med = median(parole_ratio, na.rm = TRUE),
    pct_parole = mean(has_speech_flag) * 100,
    n = n(),
    .groups = "drop"
  )

plot_box_sr <- function() {
  ggplot(df_vid_ph, aes(x = phase_lbl, y = parole_ratio, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_sr,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.0f%%", med * 100)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_percent(accuracy = 1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    labs(
      title    = "Ratio de parole par phase",
      subtitle = paste0(
        "Speech ratio (Whisper) — ", sum(med_sr$n), " vidéos"
      ),
      x = NULL, y = "Speech ratio",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d, %s avec parole",
              LBL_PHASE_SHORT[as.character(med_sr$phase_lbl)],
              med_sr$n,
              sprintf("%.0f%%", med_sr$pct_parole)),
              collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_sr(),
  file.path(OUT, "39c_speech_ratio_phase_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 39d — Overlay indexé : parole_ratio vs visages_magyar_ratio
# ---------------------------------------------------------------------------

df_clean <- df_clean |>
  ensure_col("visages_magyar_ratio", NA_real_)

df_overlay <- df_vid |>
  mutate(
    visages_magyar_ratio = suppressWarnings(
      as.numeric(df_clean$visages_magyar_ratio[match(message_id, df_clean$message_id)])
    )
  ) |>
  group_by(mois) |>
  summarise(
    sr_med    = median(parole_ratio, na.rm = TRUE),
    face_med  = median(visages_magyar_ratio, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(!is.na(sr_med), !is.na(face_med)) |>
  mutate(
    sr_idx   = to_index_100(sr_med),
    face_idx = to_index_100(face_med)
  )

if (nrow(df_overlay) >= 3 && any(!is.na(df_overlay$face_idx))) {
  pal_ov <- c(
    "Speech ratio (voix)" = unname(PAL_PHASE["2_Semi-pro"]),
    "Présence Magyar (visage)" = unname(PAL_PHASE["1_Artisanal"])
  )

  p_ov <- df_overlay |>
    select(mois, sr_idx, face_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      sr_idx   = "Speech ratio (voix)",
      face_idx = "Présence Magyar (visage)"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    scale_colour_manual(values = pal_ov) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "Dépersonnalisation : voix et visage de Magyar",
      subtitle = "Indices 0–100 (médiane mensuelle) — convergence attendue vers 0",
      x = NULL, y = "Indice (0 = min, 100 = max)",
      caption  = "Source : messages_clean.jsonl. Speech ratio (Whisper) + InsightFace."
    )

  save_plot(
    p_ov,
    file.path(OUT, "39d_overlay_voix_visage.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# Résumé
message("\n=== Speech ratio par phase ===")
print(as.data.frame(med_sr))
message("=== Terminé : exports dans ", OUT, " ===")
