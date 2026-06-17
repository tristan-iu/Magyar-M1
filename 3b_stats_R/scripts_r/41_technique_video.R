# 41_technique_video.R — Évolution des paramètres techniques vidéo
# Bitrate, FPS, taille fichier, codec
# Produit : 4_data_et_viz/41a-41f (bitrate, fps, taille, codec, overlay)
# Rscript 3b_stats_R/scripts_r/41_technique_video.R

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
message("=== 41_technique_video.R — ", nrow(df_clean), " messages chargés ===")

# Colonnes techniques (ajoutées par ffprobe_batch)
df_clean <- df_clean |>
  ensure_col("video_bitrate", NA_real_) |>
  ensure_col("fps", NA_real_) |>
  ensure_col("fichier_taille", NA_real_) |>
  ensure_col("video_codec", NA_character_) |>
  ensure_col("audio_present", NA)

df_vid <- df_clean |>
  filter(media_type == "video", !is.na(mois)) |>
  mutate(
    mois = as.Date(mois),
    video_bitrate = suppressWarnings(as.numeric(video_bitrate)),
    fps = suppressWarnings(as.numeric(fps)),
    file_size_mb = suppressWarnings(as.numeric(fichier_taille)) / (1024^2),
    video_codec = as.character(video_codec),
    audio_present = as.logical(audio_present),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  )

n_videos <- nrow(df_vid)
message("  Vidéos : ", n_videos)

# Agrégation mensuelle
df_mois_tech <- df_vid |>
  group_by(mois) |>
  summarise(
    bitrate_med = median(video_bitrate, na.rm = TRUE),
    fps_med     = median(fps, na.rm = TRUE),
    size_med_mb = median(file_size_mb, na.rm = TRUE),
    n_videos    = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

# ---------------------------------------------------------------------------
# 41a — Bitrate médian par mois
# ---------------------------------------------------------------------------

plot_bitrate <- function(df, titre = "Bitrate vidéo médian par mois") {
  ggplot(df, aes(x = mois, y = bitrate_med / 1000)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["1_Artisanal"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " kbps", accuracy = 1),
      expand = expansion(mult = c(0.02, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle — proxy de qualité d'encodage",
      x = NULL, y = "Bitrate vidéo (kbps, médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_videos,
        " vidéos. ffprobe video_bitrate."
      )
    )
}

save_plot(
  plot_bitrate(df_mois_tech),
  file.path(OUT, "41a_bitrate_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 41b — FPS médian par mois
# ---------------------------------------------------------------------------

plot_fps <- function(df, titre = "FPS médian par mois") {
  ggplot(df, aes(x = mois, y = fps_med)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["2_Semi-pro"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      expand = expansion(mult = c(0.02, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle — stabilité ou montée en gamme",
      x = NULL, y = "FPS (médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_videos,
        " vidéos. ffprobe fps."
      )
    )
}

save_plot(
  plot_fps(df_mois_tech),
  file.path(OUT, "41b_fps_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 41c — Taille fichier médiane par mois
# ---------------------------------------------------------------------------

plot_filesize <- function(df, titre = "Taille médiane des fichiers vidéo") {
  ggplot(df, aes(x = mois, y = size_med_mb)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["3_Institutionnel"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_videos), colour = PAL_PHASE["3_Institutionnel"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb vidéos") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " Mo", accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle (Mo) — ratio durée/qualité",
      x = NULL, y = "Taille fichier (Mo, médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_videos,
        " vidéos. ffprobe fichier_taille."
      )
    )
}

save_plot(
  plot_filesize(df_mois_tech),
  file.path(OUT, "41c_filesize_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 41d — Boxplots bitrate par phase
# ---------------------------------------------------------------------------

df_vid_ph <- df_vid |> filter(!is.na(phase_lbl))

med_bitrate <- df_vid_ph |>
  filter(!is.na(video_bitrate)) |>
  group_by(phase_lbl) |>
  summarise(med = median(video_bitrate / 1000, na.rm = TRUE),
            n = n(), .groups = "drop")

plot_box_bitrate <- function() {
  df_plot <- df_vid_ph |> filter(!is.na(video_bitrate))
  ggplot(df_plot, aes(x = phase_lbl, y = video_bitrate / 1000, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_bitrate,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.0f", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(suffix = " kbps"),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    coord_cartesian(ylim = c(0, quantile(df_plot$video_bitrate / 1000, 0.95, na.rm = TRUE) * 1.2)) +
    labs(
      title    = "Bitrate vidéo par phase",
      subtitle = paste0("Médiane — ", sum(med_bitrate$n), " vidéos avec données ffprobe"),
      x = NULL, y = "Bitrate (kbps)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(med_bitrate$phase_lbl)],
              med_bitrate$n), collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_bitrate(),
  file.path(OUT, "41d_bitrate_phase_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 41e — Codec vidéo : répartition par phase
# ---------------------------------------------------------------------------

df_codec <- df_vid_ph |>
  filter(!is.na(video_codec)) |>
  mutate(
    codec_grp = case_when(
      grepl("h264|avc", video_codec, ignore.case = TRUE) ~ "H.264",
      grepl("h265|hevc", video_codec, ignore.case = TRUE) ~ "H.265 (HEVC)",
      grepl("vp9", video_codec, ignore.case = TRUE) ~ "VP9",
      grepl("av1", video_codec, ignore.case = TRUE) ~ "AV1",
      TRUE ~ "Autre"
    )
  )

check_n_categories(df_codec$codec_grp, max_n = 6, var_name = "codec")

df_codec_pct <- df_codec |>
  count(phase_lbl, codec_grp) |>
  group_by(phase_lbl) |>
  mutate(pct = n / sum(n)) |>
  ungroup()

plot_codec_phase <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = pct, fill = codec_grp)) +
    geom_col(position = "fill", colour = "white", linewidth = 0.3) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_fill_cat() +
    labs(
      title    = "Codec vidéo par phase",
      subtitle = "Répartition relative — transition technologique",
      x = NULL, y = "Proportion (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        nrow(df_codec), " vidéos avec codec identifié (ffprobe)."
      )
    )
}

save_plot(
  plot_codec_phase(df_codec_pct),
  file.path(OUT, "41e_codec_phase.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 41f — Overlay indexé : bitrate + fps + durée
# ---------------------------------------------------------------------------

df_mois_ov <- df_mois_tech |>
  filter(!is.na(bitrate_med), !is.na(fps_med)) |>
  mutate(
    bitrate_idx = to_index_100(bitrate_med),
    fps_idx     = to_index_100(fps_med),
    size_idx    = to_index_100(size_med_mb)
  )

if (nrow(df_mois_ov) >= 3) {
  pal_tech <- c(
    "Bitrate"         = unname(PAL_PHASE["1_Artisanal"]),
    "FPS"             = unname(PAL_PHASE["2_Semi-pro"]),
    "Taille fichier"  = unname(PAL_PHASE["3_Institutionnel"])
  )

  p_ov <- df_mois_ov |>
    select(mois, bitrate_idx, fps_idx, size_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      bitrate_idx = "Bitrate",
      fps_idx     = "FPS",
      size_idx    = "Taille fichier"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 1.8, na.rm = TRUE) +
    scale_colour_manual(values = pal_tech) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "Paramètres techniques vidéo (indexés)",
      subtitle = "Indices min-max (0–100) — montée en gamme technique = H1",
      x = NULL, y = "Indice (0 = min, 100 = max)",
      caption  = "Source : messages_clean.jsonl. Bitrate + FPS + taille (ffprobe)."
    )

  save_plot(
    p_ov,
    file.path(OUT, "41f_technique_overlay.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# Résumé
med_tech <- df_vid_ph |>
  filter(!is.na(phase_lbl)) |>
  group_by(phase_lbl) |>
  summarise(
    n = n(),
    bitrate_med_kbps = round(median(video_bitrate / 1000, na.rm = TRUE), 0),
    fps_med          = round(median(fps, na.rm = TRUE), 1),
    size_med_mb      = round(median(file_size_mb, na.rm = TRUE), 1),
    .groups = "drop"
  )

message("\n=== Paramètres techniques par phase ===")
print(as.data.frame(med_tech))
message("=== Terminé : exports dans ", OUT, " ===")
