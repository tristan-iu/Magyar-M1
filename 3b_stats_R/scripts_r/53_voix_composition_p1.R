# 53_voix_composition_p1.R — Part de la voix en P1 : texture INA (parole / musique / silence)
# Produit : 4_data_et_viz/53a_stacked_audio_p1.png, 53b_strip_voix_p1.png,
#           53c_violin_parole_phases.png, 53d_violin_musique_phases.png,
#           53e_violin_duo_parole_musique_phases.png
# Rscript 3b_stats_R/scripts_r/53_voix_composition_p1.R

library(patchwork)

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
message("=== 53_voix_composition_p1.R — ", nrow(df_clean), " messages chargés ===")

PAL_AUDIO <- c(
  "Parole"  = unname(PAL_PHASE["1_Artisanal"]),
  "Musique" = unname(PAL_PHASE["2_Semi-pro"]),
  "Silence" = "#AAAAAA"
)

# Bornes P1 sourcées depuis r_source.R (source unique = config.yaml)
P1_DEBUT <- bornes$p1[1]
P1_FIN   <- bornes$p1[2]

# On prépare le corpus vidéo avec les champs INA
df_vid <- df_clean |>
  filter(media_type == "video", !is.na(audio_dominant), !is.na(mois)) |>
  mutate(
    mois                    = as.Date(mois),
    date                    = as.Date(date),
    audio_parole_pure_ratio = suppressWarnings(as.numeric(audio_parole_pure_ratio)),
    audio_musique_ratio     = suppressWarnings(as.numeric(audio_musique_ratio)),
    audio_silence_ratio     = suppressWarnings(as.numeric(audio_silence_ratio)),
    duree                   = suppressWarnings(as.numeric(duree)),
    phase_lbl               = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = names(PAL_PHASE)),
    audio_cat = factor(
      case_when(
        audio_dominant == "parole"  ~ "Parole",
        audio_dominant == "musique" ~ "Musique",
        TRUE                        ~ "Silence"
      ),
      levels = c("Parole", "Musique", "Silence")
    )
  )

df_p1 <- df_vid |> filter(phase == 1L)
n_p1  <- nrow(df_p1)
message("  Vidéos P1 avec segmentation INA : ", n_p1)

# ── 53a — Stacked area 100% mensuel : composition audio P1 ────────────────────
# Position fill = proportion automatique.

df_mois_comp <- df_p1 |>
  group_by(mois) |>
  summarise(
    Parole  = mean(audio_parole_pure_ratio, na.rm = TRUE),
    Musique = mean(audio_musique_ratio,     na.rm = TRUE),
    Silence = mean(audio_silence_ratio,     na.rm = TRUE),
    n       = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(as.Date(format(P1_DEBUT, "%Y-%m-01")),
                      as.Date(format(P1_FIN,   "%Y-%m-01")), by = "month")) |>
  pivot_longer(c(Parole, Musique, Silence), names_to = "categorie", values_to = "ratio") |>
  mutate(categorie = factor(categorie, levels = c("Parole", "Musique", "Silence")))

p_53a <- ggplot(df_mois_comp |> filter(!is.na(ratio)),
                aes(x = mois, y = ratio, fill = categorie)) +
  geom_area(position = "fill", alpha = 0.88) +
  scale_fill_manual(values = PAL_AUDIO, name = NULL) +
  scale_x_date(date_labels = "%b\n%Y", date_breaks = "2 months",
               expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(labels = label_percent(accuracy = 1),
                     breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = c(0, 0))) +
  labs(
    title    = "Composition audio en P1 — parole, musique, silence",
    subtitle = "Proportion mensuelle moyenne (inaSpeechSegmenter) — vidéos P1",
    x = NULL, y = NULL,
    caption  = paste0("Source : messages_clean.jsonl — ", n_p1,
                      " vidéos P1 (sept. 2022 – déc. 2023). inaSpeechSegmenter-0.7.")
  )

save_plot(p_53a, file.path(OUT, "53a_stacked_audio_p1.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ── 53b — Strip chronologique : chaque vidéo P1 = un point ───────────────────
# Chaque vidéo est un point sur la timeline. Y = part de parole (INA),
# couleur = catégorie dominante.

p_53b <- ggplot(df_p1 |> filter(!is.na(audio_parole_pure_ratio)),
                aes(x = date, y = audio_parole_pure_ratio, colour = audio_cat)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50", linewidth = 0.5) +
  annotate("text", x = P1_FIN, y = 0.52,
           label = "50 % — parole = moitié de la vidéo",
           hjust = 1, vjust = 0, size = 3, colour = "grey50") +
  geom_point(alpha = 0.65, size = 2, na.rm = TRUE) +
  # Ligne de tendance loess pour la vue d'ensemble
  geom_smooth(aes(group = 1), method = "loess", span = 0.4, se = TRUE,
              colour = "grey30", fill = "grey80", linewidth = 0.7,
              alpha = 0.25, na.rm = TRUE) +
  scale_colour_manual(values = PAL_AUDIO, name = "Dominant") +
  scale_x_date(date_labels = "%b %Y", date_breaks = "2 months",
               expand = expansion(mult = c(0.02, 0.02))) +
  scale_y_continuous(labels = label_percent(accuracy = 1),
                     limits = c(0, 1),
                     expand = expansion(mult = c(0.01, 0.03))) +
  labs(
    title    = "Part de parole par vidéo (sept. 2022 – janv. 2024)",
    subtitle = "Chaque point = une vidéo. Courbe = tendance LOESS.",
    x = NULL, y = "Part de parole (% de la durée vidéo)",
    caption  = paste0("n = ", n_p1, " vidéos")
  ) +
  guides(
    colour = guide_legend(override.aes = list(size = 3, alpha = 1))
  )

save_plot(p_53b, file.path(OUT, "53b_strip_voix_p1.png"),
          format = "wide_16_9", width = 12, dpi = 600)

# ── Helper : boxplot par phase à partir d'une colonne ratio INA ──────────────
# Le violon est inadapté ici : pour la musique, les densités P1/P2 sont écrasées
# contre y=0 et la forme disparaît derrière le bar central. On garde le boxplot
# pur + jitter discret pour la masse, et un label blanc à la médiane (toujours
# lisible, même quand la boîte est minuscule).
# Y forcé à [0, 1] pour comparabilité entre panneaux du duo.
make_box_phase <- function(df, y_col, y_label, title, subtitle) {
  df_f <- df |> filter(!is.na(.data[[y_col]]))

  med_df <- df_f |>
    group_by(phase_lbl) |>
    summarise(
      med = median(.data[[y_col]], na.rm = TRUE),
      n   = n(),
      .groups = "drop"
    )

  p <- ggplot(df_f, aes(x = phase_lbl, y = .data[[y_col]], fill = phase_lbl)) +
    geom_jitter(width = 0.12, alpha = 0.18, size = 0.6,
                colour = "grey25", show.legend = FALSE) +
    geom_boxplot(width = 0.5, alpha = 0.85, colour = "grey20",
                 outlier.shape = NA, linewidth = 0.45) +
    geom_label(data = med_df,
               aes(x = phase_lbl, y = med,
                   label = sprintf("%.0f %%", med * 100)),
               vjust = 0.5, hjust = 0.5, size = 3.6, fontface = "bold",
               fill = "white", colour = "grey15",
               label.size = 0.2, label.padding = unit(0.18, "lines"),
               inherit.aes = FALSE) +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(labels = label_percent(accuracy = 1),
                       limits = c(0, 1),
                       expand = expansion(mult = c(0.02, 0.08))) +
    labs(title = title, subtitle = subtitle,
         x = NULL, y = y_label) +
    guides(fill = "none")

  list(plot = p, med = med_df)
}

build_caption <- function(med_df) {
  paste0(
    "Source : messages_clean.jsonl. inaSpeechSegmenter-0.7. ",
    paste(sprintf("%s n=%d",
                  LBL_PHASE_SHORT[as.character(med_df$phase_lbl)],
                  med_df$n),
          collapse = " | "), "."
  )
}

# ── 53c — Boxplot : parole_ratio INA par phase (P1 vs P2 vs P3) ──────────────
res_parole <- make_box_phase(
  df_vid,
  y_col    = "audio_parole_pure_ratio",
  y_label  = "Part de parole",
  title    = "Part de parole par phase — corpus complet",
  subtitle = paste0("Distribution de audio_parole_pure_ratio (INA) — ",
                    sum(!is.na(df_vid$audio_parole_pure_ratio)), " vidéos")
)

p_53c <- res_parole$plot + labs(caption = build_caption(res_parole$med))

save_plot(p_53c, file.path(OUT, "53c_violin_parole_phases.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ── 53d — Boxplot : musique_ratio INA par phase ──────────────────────────────
# Symétrique de 53c : on regarde la part de musique.

df_vid_audio <- df_vid |> filter(!is.na(audio_musique_ratio))

res_musique <- make_box_phase(
  df_vid_audio,
  y_col    = "audio_musique_ratio",
  y_label  = "Part de musique",
  title    = "Part de musique par phase — corpus complet",
  subtitle = paste0("Distribution de audio_musique_ratio (INA) — ",
                    nrow(df_vid_audio), " vidéos")
)

p_53d <- res_musique$plot + labs(caption = build_caption(res_musique$med))

save_plot(p_53d, file.path(OUT, "53d_violin_musique_phases.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ── 53e — Duo parole | musique : deux panneaux côte-à-côte ────────────────────
# Échelle Y identique [0, 1] sur les deux panneaux.

# Reconstruit sans subtitle individuel, le wrapper patchwork porte le titre commun.
res_parole_duo <- make_box_phase(
  df_vid,
  y_col    = "audio_parole_pure_ratio",
  y_label  = "Part de parole",
  title    = "Distribution ratio parole par phase",
  subtitle = NULL
)

res_musique_duo <- make_box_phase(
  df_vid_audio,
  y_col    = "audio_musique_ratio",
  y_label  = "Part de musique",
  title    = "Distribution ratio musique par phase",
  subtitle = NULL
)

n_duo <- nrow(df_vid_audio)

DATES_PHASE <- c(
  "1_Artisanal"      = "sept. 2022 – déc. 2023",
  "2_Semi-pro"       = "janv. 2024 – sept. 2024",
  "3_Institutionnel" = "oct. 2024 – sept. 2025"
)
cap_duo_dates <- paste0(
  paste(sprintf("%s n=%d",
                DATES_PHASE[as.character(res_musique_duo$med$phase_lbl)],
                res_musique_duo$med$n),
        collapse = " | "), "."
)

p_53e <- (res_parole_duo$plot | res_musique_duo$plot) +
  plot_annotation(
    title   = paste0("Distribution par phases des ratios parole et musique, n = ",
                     n_duo, " vidéos"),
    caption = cap_duo_dates,
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 18, hjust = 0.5),
      plot.subtitle = element_text(color = "grey30", size = 13, hjust = 0.5),
      plot.caption  = element_text(color = "grey35", size = 10, hjust = 0)
    )
  )

save_plot(p_53e, file.path(OUT, "53e_violin_duo_parole_musique_phases.png"),
          format = "wide_16_9", width = 14, dpi = 600)

# Résumé console
message("\n=== Composition audio P1 (INA) — moyennes par mois ===")
df_resume <- df_p1 |>
  group_by(mois) |>
  summarise(
    n         = n(),
    parole    = round(mean(audio_parole_pure_ratio, na.rm = TRUE), 3),
    musique   = round(mean(audio_musique_ratio,     na.rm = TRUE), 3),
    silence   = round(mean(audio_silence_ratio,     na.rm = TRUE), 3),
    .groups   = "drop"
  )
print(as.data.frame(df_resume))
message("=== Terminé : 3 exports dans ", OUT, " ===")
