# 48_blasons.R — Détection des blasons de brigade (SIFT)
# Produit : 4_data_et_viz/48a_blason_mois.png, 48b_blason_types_phase.png,
#           48c_blason_roi_phase.png, 48d_blason_validation_croisee.png
# Rscript 3b_stats_R/scripts_r/48_blasons.R

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
message("=== 48_blasons.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Champs blason — directement dans df_clean (messages_clean.jsonl, schéma canonique).
# Colonnes : blason_present, blason_detecte, blason_zone.
# ---------------------------------------------------------------------------

df <- df_clean |>
  mutate(
    blason_present = coalesce(as.logical(blason_present), FALSE),
    mois = as.Date(mois)
  )

# Médias seulement (le blason n'a de sens que sur photo/vidéo)
df_media <- df |> filter(has_media, !is.na(phase))

n_media <- nrow(df_media)
message("  Médias avec champ blason : ", sum(!is.na(df_media$blason_present)))
message("  Médias avec blason détecté : ", sum(df_media$blason_present, na.rm = TRUE))

# ---------------------------------------------------------------------------
# 48a — % de médias avec blason par mois (courbe + bande de phase)
# ---------------------------------------------------------------------------

df_mois <- df_media |>
  filter(!is.na(mois)) |>
  group_by(mois) |>
  summarise(
    pct_blason = mean(blason_present, na.rm = TRUE) * 100,
    n          = n(),
    .groups    = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"),
           fill = list(pct_blason = NA_real_, n = 0L))

p48a <- ggplot(df_mois, aes(x = mois, y = pct_blason)) +
  geom_phase_bands() +
  geom_phase_lines() +
  geom_col(fill = PAL_PHASE["2_Semi-pro"], width = 25, alpha = 0.85, na.rm = TRUE) +
  geom_smooth(
    data    = filter(df_mois, !is.na(pct_blason), n >= 5),
    method  = "loess", span = 0.35, se = FALSE,
    colour  = "#1a1a1a", linewidth = 0.8, linetype = "solid"
  ) +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(
    labels = label_number(suffix = " %", accuracy = 1),
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title    = "Blason de brigade détecté par mois",
    subtitle = "% de médias (vidéos + photos) avec logo 414 OBr identifié par SIFT",
    x = NULL, y = "Médias avec blason (%)",
    caption  = paste0(
      "Source : messages_clean.jsonl — ", n_media,
      " médias. Méthode : SIFT + RANSAC, seuil 15 inliers."
    )
  )

save_plot(p48a, file.path(OUT, "48a_blason_mois.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 48b — Types de blason par phase (barplot 100% empilé)
# ---------------------------------------------------------------------------

# Ne garder que les médias avec blason détecté, puis compléter avec "Aucun"
niveaux_type <- c("414_obr", "414_mono", "pm_SARG")
lbl_type <- c(
  "414_obr"  = "414 ОБр (couleur)",
  "414_mono" = "414 ОБр (mono)",
  "pm_SARG"  = "Мадяр / САРГ"
)
pal_type <- c(
  "414 ОБр (couleur)" = "#D95F02",
  "414 ОБр (mono)"    = "#FC8D62",
  "Мадяр / САРГ"      = "#7570B3"
)

phase_lbl_levels <- c("1_Artisanal", "2_Semi-pro", "3_Institutionnel")

df_type_phase <- df_media |>
  filter(blason_present, !is.na(blason_detecte), !is.na(phase)) |>
  mutate(
    type_lbl = recode(blason_detecte, !!!lbl_type),
    type_lbl = factor(type_lbl, levels = names(pal_type)),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = phase_lbl_levels)
  ) |>
  count(phase_lbl, type_lbl) |>
  complete(phase_lbl, type_lbl = factor(names(pal_type), levels = names(pal_type)),
           fill = list(n = 0L))

n_par_phase <- df_type_phase |>
  group_by(phase_lbl) |>
  summarise(total = sum(n), .groups = "drop")

p48b <- ggplot(df_type_phase, aes(x = phase_lbl, y = n, fill = type_lbl)) +
  geom_col(position = "fill", colour = "white", linewidth = 0.25) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  scale_fill_manual(values = pal_type, name = NULL) +
  labs(
    title    = "Types de blason détectés par phase",
    subtitle = "Part relative des variantes du logo 414 OBr (SIFT, médias avec blason seulement)",
    x = NULL, y = "Proportion (%)",
    caption  = paste0(
      "Source : messages_clean.jsonl. ",
      paste(sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(n_par_phase$phase_lbl)],
                    n_par_phase$total), collapse = " | "), "."
    )
  ) +
  theme(legend.position = "bottom")

save_plot(p48b, file.path(OUT, "48b_blason_types_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 48c — Position du blason (ROI) par phase
# ---------------------------------------------------------------------------

niveaux_roi <- c("haut_droite", "bas_droite", "image_entiere")
lbl_roi <- c(
  "haut_droite"   = "Haut droite",
  "bas_droite"    = "Bas droite",
  "image_entiere" = "Image entière (photo)"
)
pal_roi <- c(
  "Haut droite"           = "#1B9E77",
  "Bas droite"            = "#D95F02",
  "Image entière (photo)" = "#7570B3"
)

df_roi_phase <- df_media |>
  filter(blason_present, !is.na(blason_zone), !is.na(phase)) |>
  mutate(
    roi_lbl = recode(blason_zone, !!!lbl_roi),
    roi_lbl = factor(roi_lbl, levels = names(pal_roi)),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = phase_lbl_levels)
  ) |>
  count(phase_lbl, roi_lbl) |>
  complete(phase_lbl, roi_lbl = factor(names(pal_roi), levels = names(pal_roi)),
           fill = list(n = 0L))

n_roi_phase <- df_roi_phase |>
  group_by(phase_lbl) |>
  summarise(total = sum(n), .groups = "drop")

p48c <- ggplot(df_roi_phase, aes(x = phase_lbl, y = n, fill = roi_lbl)) +
  geom_col(position = "fill", colour = "white", linewidth = 0.25) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  scale_fill_manual(values = pal_roi, name = NULL) +
  labs(
    title    = "Position du blason dans l'image par phase",
    subtitle = "Standardisation progressive du placement — P3 : quasi exclusivement haut droite",
    x = NULL, y = "Proportion (%)",
    caption  = paste0(
      "Source : messages_clean.jsonl. ",
      paste(sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(n_roi_phase$phase_lbl)],
                    n_roi_phase$total), collapse = " | "), "."
    )
  ) +
  theme(legend.position = "bottom")

save_plot(p48c, file.path(OUT, "48c_blason_roi_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 48d — Validation croisée : blason SIFT vs OCR "414 ОБр"
# Deux méthodes indépendantes (vision + texte)
# ---------------------------------------------------------------------------

df_valid <- df |>
  filter(has_media, !is.na(mois)) |>
  mutate(
    has_blason_sift = blason_present,
    # OCR : présence de "414" dans le texte détecté (repris du script 42)
    ocr_all = paste0(
      replace_na(as.character(ocr_texte), ""), " ",
      replace_na(as.character(ocr_filigrane_texte), "")
    ),
    has_414_ocr = grepl("414", ocr_all)
  ) |>
  group_by(mois) |>
  summarise(
    pct_sift = mean(has_blason_sift, na.rm = TRUE) * 100,
    pct_ocr  = mean(has_414_ocr,    na.rm = TRUE) * 100,
    n        = n(),
    .groups  = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"),
           fill = list(pct_sift = NA_real_, pct_ocr = NA_real_, n = 0L)) |>
  pivot_longer(c(pct_sift, pct_ocr),
               names_to = "methode", values_to = "pct") |>
  mutate(methode = recode(methode,
    pct_sift = "SIFT (logo visuel)",
    pct_ocr  = "OCR (texte «414»)"
  ))

pal_methode <- c(
  "SIFT (logo visuel)" = "#D95F02",
  "OCR (texte «414»)"  = "#1B9E77"
)

p48d <- ggplot(df_valid, aes(x = mois, y = pct, colour = methode, group = methode)) +
  geom_phase_bands() +
  geom_phase_lines() +
  geom_line(linewidth = 0.9, na.rm = TRUE) +
  geom_point(data = filter(df_valid, !is.na(pct), n >= 5),
             size = 1.5, na.rm = TRUE) +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(
    labels = label_number(suffix = " %", accuracy = 1),
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.04))
  ) +
  scale_colour_manual(values = pal_methode, name = NULL) +
  labs(
    title    = "Convergence des deux signaux de branding «414 OBr»",
    subtitle = "SIFT (reconnaissance de logo) vs OCR (détection du texte) — mêmes données, méthodes indépendantes",
    x = NULL, y = "Médias avec signal détecté (%)",
    caption  = paste0(
      "Source : messages_clean.jsonl — ", n_media, " médias. ",
      "SIFT : seuil 15 inliers RANSAC. OCR : regex «414» dans ocr_texte / ocr_filigrane_texte."
    )
  ) +
  theme(legend.position = "bottom")

save_plot(p48d, file.path(OUT, "48d_blason_validation_croisee.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# Résumé console
# ---------------------------------------------------------------------------

message("\n=== Blason par phase (médias) ===")
df_media |>
  filter(!is.na(phase)) |>
  group_by(phase) |>
  summarise(
    n_total  = n(),
    n_blason = sum(blason_present, na.rm = TRUE),
    pct      = round(100 * n_blason / n_total, 1),
    .groups  = "drop"
  ) |>
  print()

message("\n=== Catégories de blason ===")
df_media |>
  filter(blason_present, !is.na(blason_detecte)) |>
  count(blason_detecte, sort = TRUE) |>
  print()

message("=== Terminé : exports dans ", OUT, " ===")
