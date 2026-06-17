# 51_audio_segmentation.R — Segmentation audio INA (parole / musique / silence)
# Produit : 4_data_et_viz/51a_audio_composition_phase.png, 51b_musique_ratio_boxplot.png
# Rscript 3b_stats_R/scripts_r/51_audio_segmentation.R

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
message("=== 51_audio_segmentation.R — ", nrow(df_clean), " messages chargés ===")

# On s'assure que les champs audio_* existent (absents pour non-audio)
df_clean <- df_clean |>
  ensure_col("audio_parole_pure_ratio", NA_real_) |>
  ensure_col("audio_musique_ratio",     NA_real_) |>
  ensure_col("audio_silence_ratio",     NA_real_) |>
  ensure_col("audio_dominant",          NA_character_) |>
  ensure_col("alerte_musique_dominante", NA)

# Corpus : vidéos avec segmentation audio disponible
df_audio <- df_clean |>
  filter(media_type == "video", !is.na(audio_dominant), !is.na(phase)) |>
  mutate(
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = names(PAL_PHASE)),
    audio_parole_pure_ratio = suppressWarnings(as.numeric(audio_parole_pure_ratio)),
    audio_musique_ratio     = suppressWarnings(as.numeric(audio_musique_ratio)),
    audio_silence_ratio     = suppressWarnings(as.numeric(audio_silence_ratio))
  )

n_audio <- nrow(df_audio)
message("  Vidéos avec segmentation audio : ", n_audio)

# ---------------------------------------------------------------------------
# 51a — Composition audio moyenne par phase (barres empilées)
# ---------------------------------------------------------------------------

# Remappage intentionnel : Parole/Musique/Silence ne sont PAS des phases.
# On réutilise les couleurs de PAL_PHASE (r_source.R) par cohérence visuelle ;
# Silence = gris neutre, hors palette de phase.
PAL_AUDIO <- c(
  "Parole"  = unname(PAL_PHASE["1_Artisanal"]),
  "Musique" = unname(PAL_PHASE["2_Semi-pro"]),
  "Silence" = "#AAAAAA"
)

df_comp <- df_audio |>
  group_by(phase_lbl) |>
  summarise(
    Parole  = mean(audio_parole_pure_ratio, na.rm = TRUE),
    Musique = mean(audio_musique_ratio,     na.rm = TRUE),
    Silence = mean(audio_silence_ratio,     na.rm = TRUE),
    n       = n(),
    .groups = "drop"
  ) |>
  pivot_longer(c(Parole, Musique, Silence), names_to = "categorie", values_to = "ratio") |>
  mutate(categorie = factor(categorie, levels = c("Parole", "Musique", "Silence")))

p_comp <- ggplot(df_comp, aes(x = phase_lbl, y = ratio, fill = categorie)) +
  geom_col(width = 0.6, position = "stack") +
  geom_text(
    aes(label = sprintf("%.0f%%", ratio * 100)),
    position = position_stack(vjust = 0.5),
    colour = "white", fontface = "bold", size = 4
  ) +
  scale_fill_manual(values = PAL_AUDIO, name = NULL) +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  scale_y_continuous(
    labels = label_percent(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title    = "Composition audio par phase",
    subtitle = "Proportion moyenne de parole pure, musique et silence (inaSpeechSegmenter)",
    x = NULL, y = "Part moyenne",
    caption  = paste0(
      "Source : messages_clean.jsonl — ", n_audio, " vidéos avec segmentation audio. ",
      paste(
        sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(unique(df_comp$phase_lbl))],
                df_comp |> distinct(phase_lbl, n) |> pull(n)),
        collapse = " | "
      ), "."
    )
  ) +
  guides(fill = guide_legend(reverse = FALSE))

save_plot(p_comp, file.path(OUT, "51a_audio_composition_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 51b — Boxplot audio_musique_ratio par phase
# ---------------------------------------------------------------------------

med_mus <- df_audio |>
  group_by(phase_lbl) |>
  summarise(
    med = median(audio_musique_ratio, na.rm = TRUE),
    n   = n(),
    .groups = "drop"
  )

p_box <- ggplot(df_audio, aes(x = phase_lbl, y = audio_musique_ratio, fill = phase_lbl)) +
  geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
  geom_text(
    data = med_mus,
    aes(x = phase_lbl, y = med, label = sprintf("méd. = %.0f%%", med * 100)),
    vjust = -0.8, size = 3.8, fontface = "bold", inherit.aes = FALSE
  ) +
  scale_fill_phase(short = TRUE) +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  scale_y_continuous(
    labels = label_percent(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    title    = "Ratio de musique par phase",
    subtitle = paste0("Distribution de audio_musique_ratio — ", n_audio, " vidéos"),
    x = NULL, y = "Part musique",
    caption  = paste0(
      "Source : messages_clean.jsonl. ",
      paste(sprintf("%s : n=%d, méd.=%.0f%%",
                    LBL_PHASE_SHORT[as.character(med_mus$phase_lbl)],
                    med_mus$n,
                    med_mus$med * 100),
            collapse = " | "), "."
    )
  ) +
  guides(fill = "none")

save_plot(p_box, file.path(OUT, "51b_musique_ratio_boxplot.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# Résumé console
message("\n=== Composition audio par phase ===")
print(as.data.frame(df_comp |> select(phase_lbl, categorie, ratio) |>
  mutate(ratio = sprintf("%.1f%%", ratio * 100))))
message("=== Terminé : exports dans ", OUT, " ===")
