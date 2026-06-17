# 32_insightface.R — Présence de Magyar, individus distincts, position temporelle
# Produit : 4_data_et_viz/32a-32f (présence mensuelle, ratios par phase, stacked, vues × parole)
# Rscript 3b_stats_R/scripts_r/32_insightface.R

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(lubridate)
  library(scales)
  library(readr)
  library(purrr)
  library(forcats)
})

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

# CSV per-face (detect_magyar.py)
FACES_CSV <- file.path(BASE, "..", "2d_vision", "visages", "results", "magyar_detection.csv")

# Palette phases
pal_phase_lbl <- c(
  "P1 Artisanal"      = unname(PAL_PHASE["1_Artisanal"]),
  "P2 Semi-pro"       = unname(PAL_PHASE["2_Semi-pro"]),
  "P3 Institutionnel" = unname(PAL_PHASE["3_Institutionnel"])
)

phase_factor <- function(p) {
  factor(p, levels = c(1L, 2L, 3L),
         labels = c("P1 Artisanal", "P2 Semi-pro", "P3 Institutionnel"))
}

message("=== 32_insightface.R — ", nrow(df_clean), " messages chargés (df_clean) ===")

# ── df_f depuis df_clean (JSONL) ─────────────────────────────────────────────

df_f <- df_clean %>%
  filter(!is.na(visages_densite)) %>%
  mutate(
    mois                = as.Date(floor_date(date, "month")),
    phase_lbl           = phase_factor(phase),
    visages_densite     = as.numeric(visages_densite),
    visages_magyar_ratio  = as.numeric(visages_magyar_ratio),
    faces_max_sim       = as.numeric(visages_magyar_similarite_max),
    faces_present       = as.logical((visages_magyar_detections > 0)),
    faces_unique        = as.integer(visages_unique),
    faces_magyar_det    = as.integer(visages_magyar_detections),
    faces_non_magyar    = pmax(faces_unique - as.integer(faces_present), 0L)
  ) %>%
  filter(!is.na(phase))

message(nrow(df_f), " messages avec données faces, ",
        sum(df_f$faces_present, na.rm = TRUE), " avec Magyar détecté")

# ── CSV per-face ─────────────────────────────────────────────────────────────

message("Chargement CSV per-face : ", FACES_CSV)
df_csv <- read_csv(FACES_CSV, show_col_types = FALSE) %>%
  filter(phase != "" & !is.na(phase)) %>%
  mutate(
    date  = parse_date_safe(date),
    mois  = as.Date(floor_date(date, "month")),
    phase_int = case_when(
      phase == "P1" ~ 1L,
      phase == "P2" ~ 2L,
      phase == "P3" ~ 3L,
      TRUE ~ NA_integer_),
    phase_lbl   = phase_factor(phase_int),
    is_face     = face_index >= 0,
    is_magyar   = as.logical(is_magyar)
  ) %>%
  filter(!is.na(phase_int))

df_faces <- df_csv %>% filter(is_face)
message(nrow(df_csv), " lignes CSV, dont ", nrow(df_faces), " détections faciales")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Présence mensuelle de Magyar (% médias)
# ─────────────────────────────────────────────────────────────────────────────

monthly_presence <- df_f %>%
  group_by(mois) %>%
  summarise(
    n_total   = n(),
    n_present = sum(faces_present, na.rm = TRUE),
    pct       = 100 * n_present / n_total,
    .groups   = "drop"
  ) %>%
  filter(n_total >= 3)

cpts_presence <- compute_cpts(monthly_presence$pct, monthly_presence$mois)
message("Changepoints présence Magyar : ", paste(format(cpts_presence), collapse = ", "))

p1 <- ggplot(monthly_presence, aes(x = mois, y = pct)) +
  geom_phase_bands() +
  geom_line(colour = "grey30", linewidth = 0.9) +
  geom_point(aes(size = n_total), colour = "grey20", fill = "white",
             shape = 21, stroke = 0.8) +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, NA),
                     expand = expansion(mult = c(0, 0.05))) +
  scale_size_continuous(range = c(2, 6), name = "N médias/mois") +
  labs(
    title    = "Présence de Magyar à l'écran",
    subtitle = "% de médias avec Magyar détecté par mois — ruptures PELT/MBIC",
    x = NULL, y = "% médias avec Magyar",
    caption  = "InsightFace buffalo_l — seuil cosine 0.4 — 38 photos de référence"
  )

p1 <- add_cpt_lines(p1, cpts_presence)
save_plot(p1, file.path(OUT, "32a_magyar_presence_mensuelle.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Distribution bimodale de visages_magyar_ratio par phase
# ─────────────────────────────────────────────────────────────────────────────

df_ratio <- df_f %>%
  filter(!is.na(phase_lbl), !is.na(visages_magyar_ratio)) %>%
  mutate(
    ratio = as.numeric(visages_magyar_ratio),
    # Catégorie : absent (0), partiel (0–0.5), dominant (>0.5)
    cat = case_when(
      ratio == 0       ~ "Absent\n(0%)",
      ratio <= 0.5     ~ "Partiel\n(1–50%)",
      TRUE             ~ "Dominant\n(>50%)"
    ),
    cat = factor(cat, levels = c("Absent\n(0%)", "Partiel\n(1–50%)", "Dominant\n(>50%)"))
  )

# Comptage par phase × catégorie
ratio_counts <- df_ratio %>%
  count(phase_lbl, cat) %>%
  group_by(phase_lbl) %>%
  mutate(
    total = sum(n),
    pct   = 100 * n / total,
    label = sprintf("%d\n(%.0f%%)", n, pct)
  ) %>%
  ungroup()

p2 <- ggplot(ratio_counts, aes(x = cat, y = pct, fill = phase_lbl)) +
  geom_col(width = 0.65, alpha = 0.85) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.2,
            fontface = "bold", colour = "grey20", lineheight = 0.85) +
  facet_wrap(~phase_lbl, nrow = 1,
             labeller = as_labeller(c(
               "P1 Artisanal"      = "P1",
               "P2 Semi-pro"       = "P2",
               "P3 Institutionnel" = "P3"
             ))) +
  scale_fill_manual(values = pal_phase_lbl) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Magyar dans la publication : absent, partiel ou dominant ?",
    subtitle = "Distribution des publications selon la part de captures d'écrans avec Magyar détecté",
    x = NULL, y = "% des publications",
    caption  = "Détection par similarité de visage (38 captures de référence, seuil cosinus 0,4)."
  ) +
  theme_madyar_facet() +
  theme(legend.position = "none")

save_plot(p2, file.path(OUT, "32b_ratio_par_phase.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Quand Magyar est présent : quel ratio occupe-t-il ?
# Violin + boxplot de visages_magyar_ratio parmi les posts avec Magyar > 0.
# ─────────────────────────────────────────────────────────────────────────────

df_present <- df_ratio %>%
  filter(ratio > 0)

med_ratio <- df_present %>%
  group_by(phase_lbl) %>%
  summarise(med = median(ratio), n = n(), .groups = "drop") %>%
  mutate(label = sprintf("méd. %.0f%%\nn=%d", med * 100, n))

p3 <- ggplot(df_present, aes(x = phase_lbl, y = ratio, fill = phase_lbl)) +
  geom_violin(alpha = 0.35, colour = NA, trim = FALSE) +
  geom_boxplot(width = 0.2, outlier.size = 0.8, outlier.alpha = 0.4,
               linewidth = 0.5, fill = "white", colour = "grey30") +
  geom_text(data = med_ratio,
            aes(x = phase_lbl, y = med, label = label),
            hjust = -0.6, size = 3.2, fontface = "bold", colour = "grey20",
            lineheight = 0.85) +
  scale_fill_manual(values = pal_phase_lbl) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1.05),
                     expand = expansion(mult = c(0.02, 0))) +
  labs(
    title    = "Intensité de présence de Magyar (posts avec Magyar uniquement)",
    subtitle = "Part des keyframes avec Magyar détecté — parmi les publications où il apparaît",
    x = NULL, y = "Ratio keyframes Magyar / total",
    caption  = "En P3, quand Magyar apparaît, il occupe la quasi-totalité du post"
  ) +
  theme(legend.position = "none")

save_plot(p3, file.path(OUT, "32c_ratio_intensite_magyar.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS 2bis / 3bis — restriction P1+P2 (sans P3) — pour analyses kamikaze
# Mêmes graphiques que 32b/32c sur le sous-corpus P1+P2 (analyse kamikaze).
# ─────────────────────────────────────────────────────────────────────────────

df_ratio_p12 <- df_ratio %>%
  filter(phase %in% c(1L, 2L)) %>%
  mutate(phase_lbl = droplevels(phase_lbl))

ratio_counts_p12 <- df_ratio_p12 %>%
  count(phase_lbl, cat) %>%
  group_by(phase_lbl) %>%
  mutate(
    total = sum(n),
    pct   = 100 * n / total,
    label = sprintf("%d\n(%.0f%%)", n, pct)
  ) %>%
  ungroup()

p2_p12 <- ggplot(ratio_counts_p12, aes(x = cat, y = pct, fill = phase_lbl)) +
  geom_col(width = 0.65, alpha = 0.85) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.2,
            fontface = "bold", colour = "grey20", lineheight = 0.85) +
  facet_wrap(~phase_lbl, nrow = 1) +
  scale_fill_manual(values = pal_phase_lbl) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Magyar dans la publication : absent, partiel ou dominant ? — P1+P2",
    subtitle = "Distribution des publications selon la part de keyframes avec Magyar détecté",
    x = NULL, y = "% des publications",
    caption  = paste0(
      "InsightFace buffalo_l — seuil cosine 0.4 — 38 photos de référence. ",
      "Sous-corpus P1+P2 (n=", nrow(df_ratio_p12), "), P3 exclue."
    )
  ) +
  theme_madyar_facet() +
  theme(legend.position = "none")

save_plot(p2_p12, file.path(OUT, "32b_ratio_par_phase_p12.png"), format = "wide_16_9")

df_present_p12 <- df_ratio_p12 %>% filter(ratio > 0)

med_ratio_p12 <- df_present_p12 %>%
  group_by(phase_lbl) %>%
  summarise(med = median(ratio), n = n(), .groups = "drop") %>%
  mutate(label = sprintf("méd. %.0f%%\nn=%d", med * 100, n))

p3_p12 <- ggplot(df_present_p12, aes(x = phase_lbl, y = ratio, fill = phase_lbl)) +
  geom_violin(alpha = 0.35, colour = NA, trim = FALSE) +
  geom_boxplot(width = 0.2, outlier.size = 0.8, outlier.alpha = 0.4,
               linewidth = 0.5, fill = "white", colour = "grey30") +
  geom_text(data = med_ratio_p12,
            aes(x = phase_lbl, y = med, label = label),
            hjust = -0.6, size = 3.2, fontface = "bold", colour = "grey20",
            lineheight = 0.85) +
  scale_fill_manual(values = pal_phase_lbl) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1.05),
                     expand = expansion(mult = c(0.02, 0))) +
  labs(
    title    = "Intensité de présence de Magyar — P1+P2 (posts avec Magyar uniquement)",
    subtitle = "Part des keyframes avec Magyar détecté — parmi les publications où il apparaît",
    x = NULL, y = "Ratio keyframes Magyar / total",
    caption  = paste0(
      "Sous-corpus P1+P2 (n=", nrow(df_present_p12), " posts avec Magyar), P3 exclue."
    )
  ) +
  theme(legend.position = "none")

save_plot(p3_p12, file.path(OUT, "32c_ratio_intensite_magyar_p12.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Synthèse : absence vs présence forte par phase (stacked)
# Vue d'ensemble : % absent / partiel / dominant — facettes empilées
# ─────────────────────────────────────────────────────────────────────────────

p4 <- ggplot(ratio_counts,
             aes(x = phase_lbl, y = pct, fill = cat)) +
  geom_col(width = 0.6, alpha = 0.88) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5),
            size = 3.8, fontface = "bold", colour = "white") +
  scale_fill_manual(
    values = c("Absent\n(0%)" = "grey65",
               "Partiel\n(1–50%)" = "#E8A87C",
               "Dominant\n(>50%)" = "#C0392B"),
    name = NULL
  ) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Évolution du type de présence de Magyar par phase",
    subtitle = "Absent / Partiel (1–50% des keyframes) / Dominant (>50%)",
    x = NULL, y = "% des publications",
    caption  = "Polarisation croissante : en P3, Magyar est absent ou dominant — jamais intermédiaire"
  )

save_plot(p4, file.path(OUT, "32d_presence_type_stacked.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# Résumé chiffré
# ─────────────────────────────────────────────────────────────────────────────

message("\n=== Résumé visages_magyar_ratio par phase ===")
df_ratio %>%
  group_by(phase_lbl) %>%
  summarise(
    n_total   = n(),
    n_absent  = sum(ratio == 0),
    n_partiel = sum(ratio > 0 & ratio <= 0.5),
    n_domin   = sum(ratio > 0.5),
    pct_absent = round(100 * n_absent / n_total, 1),
    pct_domin  = round(100 * n_domin / n_total, 1),
    med_ratio_nonzero = round(median(ratio[ratio > 0], na.rm = TRUE), 3),
    .groups = "drop"
  ) %>%
  { message(paste(capture.output(print(as.data.frame(.))), collapse = "\n")) }

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS 5-6 — Vues × présence vocale (parole_ratio)
# Miroir des analyses visages (plots 2-4), sur le sous-corpus vidéo transcrit
# ─────────────────────────────────────────────────────────────────────────────

df_parole <- df_clean %>%
  filter(media_type == "video", vues > 0) %>%
  mutate(
    phase_lbl    = phase_factor(phase),
    parole_ratio = suppressWarnings(as.numeric(parole_ratio)),
    cat_parole   = case_when(
      parole_ratio == 0   ~ "Absent\n(0%)",
      parole_ratio <= 0.5 ~ "Partiel\n(1–50%)",
      TRUE                ~ "Dominant\n(>50%)"
    ),
    cat_parole = factor(cat_parole,
                        levels = c("Absent\n(0%)", "Partiel\n(1–50%)", "Dominant\n(>50%)"))
  ) %>%
  filter(!is.na(phase_lbl))

n_vid_parole <- nrow(df_parole)
message("\n=== Analyse voix — ", n_vid_parole, " vidéos ===")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Vues par catégorie parole × phase (violin + boxplot)
# ─────────────────────────────────────────────────────────────────────────────

fmt_vues <- label_number(scale_cut = cut_short_scale())

med_par <- df_parole %>%
  group_by(phase_lbl, cat_parole) %>%
  summarise(med = median(vues, na.rm = TRUE), n = n(), .groups = "drop") %>%
  mutate(label = sprintf("méd. %s\nn=%d", fmt_vues(med), n))

p5 <- ggplot(df_parole, aes(x = cat_parole, y = vues, fill = phase_lbl)) +
  geom_violin(alpha = 0.30, colour = NA, trim = FALSE) +
  geom_boxplot(width = 0.25, outlier.size = 0.7, outlier.alpha = 0.3,
               linewidth = 0.45, fill = "white", colour = "grey30") +
  geom_text(data = med_par,
            aes(x = cat_parole, y = med, label = label),
            hjust = -0.45, size = 2.9, fontface = "bold", colour = "grey20",
            lineheight = 0.85) +
  facet_wrap(~phase_lbl, nrow = 1) +
  scale_fill_manual(values = pal_phase_lbl) +
  scale_y_log10(labels = fmt_vues, expand = expansion(mult = c(0.02, 0.30))) +
  labs(
    title    = "Vues selon la présence de parole (vidéos)",
    subtitle = "Distribution des vues par intensité de parole_ratio — échelle log",
    x = NULL, y = "Vues (log)",
    caption  = paste0(
      "Whisper large-v3 + Silero VAD — ", n_vid_parole,
      " vidéos. parole_ratio = part de la durée avec parole détectée."
    )
  ) +
  theme_madyar_facet() +
  theme(legend.position = "none")

save_plot(p5, file.path(OUT, "32e_vues_parole_par_phase.png"), format = "wide_16_9")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Distribution voix par phase (stacked, miroir de 32d)
# ─────────────────────────────────────────────────────────────────────────────

parole_counts <- df_parole %>%
  count(phase_lbl, cat_parole) %>%
  group_by(phase_lbl) %>%
  mutate(total = sum(n), pct = 100 * n / total) %>%
  ungroup()

p6 <- ggplot(parole_counts, aes(x = phase_lbl, y = pct, fill = cat_parole)) +
  geom_col(width = 0.6, alpha = 0.88) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5),
            size = 3.8, fontface = "bold", colour = "white") +
  scale_fill_manual(
    values = c(
      "Absent\n(0%)"     = "grey65",
      "Partiel\n(1–50%)" = "#66C2A5",
      "Dominant\n(>50%)" = "#238B45"
    ),
    name = NULL
  ) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Évolution de la présence vocale par phase",
    subtitle = "Absent / Partiel (1–50% de la durée) / Dominant (>50%) — vidéos uniquement",
    x = NULL, y = "% des vidéos",
    caption  = paste0(
      "Silero VAD — ", n_vid_parole,
      " vidéos. Miroir de la figure 32d (présence de Magyar à l'écran)."
    )
  )

save_plot(p6, file.path(OUT, "32f_parole_distribution_stacked.png"), format = "wide_16_9")

# Résumé chiffré voix
message("\n=== Résumé parole_ratio par phase ===")
df_parole %>%
  group_by(phase_lbl) %>%
  summarise(
    n_total      = n(),
    pct_absent   = round(100 * mean(parole_ratio == 0), 1),
    pct_dominant = round(100 * mean(parole_ratio > 0.5), 1),
    med_vues_absent   = round(median(vues[parole_ratio == 0],   na.rm = TRUE)),
    med_vues_dominant = round(median(vues[parole_ratio > 0.5],  na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  { message(paste(capture.output(print(as.data.frame(.))), collapse = "\n")) }

message("\n=== Terminé — fichiers dans ", OUT, " ===")
