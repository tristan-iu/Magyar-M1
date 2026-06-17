# 49_blasons_stats.R — Analyses statistiques blasons
# Chi2 (présence × phase, blason × Magyar), AFC roi × phase, heatmap spatiale ROI
# Produit : 4_data_et_viz/49a_chi2_blason_phase.png, 49b_chi2_blason_magyar.png,
#           49c_afc_roi_phase.png, 49d_heatmap_roi_spatial.png
# Rscript 3b_stats_R/scripts_r/49_blasons_stats.R

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
message("=== 49_blasons_stats.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Préparation
# df_clean vient de messages_clean.jsonl via r_source.R
# ---------------------------------------------------------------------------

phase_lbl_levels <- c("1_Artisanal", "2_Semi-pro", "3_Institutionnel")

df <- df_clean |>
  ensure_col("blason_present", FALSE) |>
  ensure_col("blason_detecte", NA_character_) |>
  ensure_col("blason_zone",     NA_character_) |>
  ensure_col("visages_magyar_detections", NA_integer_) |>
  mutate(
    blason_present = coalesce(as.logical(blason_present), FALSE),
    magyar_present = coalesce(suppressWarnings(as.integer(visages_magyar_detections)) > 0L, FALSE),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = phase_lbl_levels)
  )

df_media <- df |> filter(has_media, !is.na(phase))
n_media   <- nrow(df_media)

message("  Médias : ", n_media,
        " | avec blason : ", sum(df_media$blason_present, na.rm = TRUE),
        " | avec donnees InsightFace : ",
        sum(!is.na(df_media$visages_magyar_detections)))

# ---------------------------------------------------------------------------
# 49a — Chi2 blason_present × phase
# Teste si la distribution du branding change significativement entre phases
# Résidus de Pearson : quelle phase sur/sous-représente le blason ?
# ---------------------------------------------------------------------------

ct_phase <- table(
  phase  = df_media$phase_lbl,
  blason = df_media$blason_present
)
chi_phase <- chisq.test(ct_phase, correct = FALSE)

message("\n=== 49a — Chi2 blason présence × phase ===")
print(ct_phase)
message(sprintf("chi2(%d) = %.2f, p = %s",
  chi_phase$parameter, chi_phase$statistic,
  formatC(chi_phase$p.value, format = "e", digits = 2)))

# Résidus standardisés — valeur absolue > 2 = association significative
resid_df <- as.data.frame(chi_phase$stdres) |>
  rename(phase = phase, blason = blason, stdres = Freq) |>
  filter(blason == "TRUE") |>
  mutate(
    direction = if_else(stdres > 0, "Sur-représenté", "Sous-représenté"),
    phase_short = LBL_PHASE_SHORT[as.character(phase)]
  )

p49a <- ggplot(resid_df, aes(x = phase_short, y = stdres, fill = direction)) +
  geom_col(colour = "white", linewidth = 0.3, width = 0.55) +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed",
             colour = "grey40", linewidth = 0.5) +
  geom_hline(yintercept = 0, colour = "grey20", linewidth = 0.4) +
  annotate("text", x = 0.6, y = 2.2, label = "seuil α = 0.05",
           hjust = 0, size = 3, colour = "grey40", fontface = "italic") +
  scale_fill_manual(
    values = c("Sur-représenté"   = unname(PAL_PHASE["2_Semi-pro"]),
               "Sous-représenté"  = unname(PAL_PHASE["3_Institutionnel"])),
    name = NULL
  ) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  labs(
    title    = "Association blason × phase — résidus de Pearson standardisés",
    subtitle = sprintf("χ²(%d) = %.1f, p < 2.2×10⁻¹⁶ — présence du logo très significativement liée à la phase",
                       chi_phase$parameter, chi_phase$statistic),
    x = NULL, y = "Résidu standardisé (cases « blason détecté »)",
    caption  = paste0(
      "Source : messages_clean.jsonl — ", n_media, " médias. ",
      "Test χ² Pearson sans correction de continuité. Seuil |r| > 2 (p < 0.05)."
    )
  ) +
  theme(legend.position = "bottom")

save_plot(p49a, file.path(OUT, "49a_chi2_blason_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 49b — Chi2 blason × Magyar présent
# ---------------------------------------------------------------------------

df_b <- df_media |>
  filter(!is.na(visages_magyar_detections))

ct_mag <- table(
  blason = df_b$blason_present,
  magyar = df_b$magyar_present
)

chi_mag <- chisq.test(ct_mag, correct = FALSE)

# Odds ratio (2×2 orienté : blason=TRUE, magyar=TRUE en [2,2])
a <- ct_mag["TRUE",  "TRUE"]
b <- ct_mag["TRUE",  "FALSE"]
c <- ct_mag["FALSE", "TRUE"]
d <- ct_mag["FALSE", "FALSE"]
or_val  <- (a * d) / (b * c)
se_log  <- sqrt(1/a + 1/b + 1/c + 1/d)
ci_or   <- exp(log(or_val) + c(-1, 1) * 1.96 * se_log)

message("\n=== 49b — Chi2 blason × Magyar présent ===")
print(ct_mag)
message(sprintf("chi2(%d) = %.2f, p = %s | OR = %.3f [%.3f–%.3f]",
  chi_mag$parameter, chi_mag$statistic,
  formatC(chi_mag$p.value, format = "e", digits = 2),
  or_val, ci_or[1], ci_or[2]))

pct_b <- df_b |>
  group_by(magyar = magyar_present) |>
  summarise(
    n          = n(),
    n_blason   = sum(blason_present, na.rm = TRUE),
    pct_blason = n_blason / n * 100,
    se         = sqrt(pct_blason * (100 - pct_blason) / n),
    .groups = "drop"
  ) |>
  mutate(
    lbl = if_else(magyar, "Magyar visible\n(InsightFace)", "Magyar absent\n(InsightFace)"),
    lbl = factor(lbl, levels = c("Magyar visible\n(InsightFace)", "Magyar absent\n(InsightFace)"))
  )

p49b <- ggplot(pct_b, aes(x = lbl, y = pct_blason, fill = lbl)) +
  geom_col(width = 0.5, colour = "white", linewidth = 0.3) +
  geom_errorbar(
    aes(ymin = pct_blason - 1.96 * se, ymax = pct_blason + 1.96 * se),
    width = 0.14, colour = "grey30", linewidth = 0.6
  ) +
  geom_text(
    aes(label = sprintf("%.1f%%\n(n=%d)", pct_blason, n),
        y = pct_blason + 1.96 * se + 1.5),
    size = 3.5, colour = "grey15"
  ) +
  scale_fill_manual(
    values = c(
      "Magyar visible\n(InsightFace)" = PAL_PHASE["1_Artisanal"],
      "Magyar absent\n(InsightFace)"  = PAL_PHASE["3_Institutionnel"]
    ),
    guide = "none"
  ) +
  scale_y_continuous(
    labels = scales::label_number(suffix = " %", accuracy = 1),
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.15))
  ) +
  labs(
    title    = "Blason et présence de Magyar : substitution symbolique",
    subtitle = sprintf(
      "χ²(%d) = %.1f, p < 0.001 | OR = %.2f [IC95%% %.2f–%.2f] — relation inverse significative",
      chi_mag$parameter, chi_mag$statistic, or_val, ci_or[1], ci_or[2]
    ),
    x = NULL, y = "Médias avec blason détecté (%)",
    caption  = paste0(
      "Source : messages_clean.jsonl × InsightFace v2 — ",
      nrow(df_b), " médias avec données de détection faciale. ",
      "Barres d'erreur : IC 95%."
    )
  )

save_plot(p49b, file.path(OUT, "49b_chi2_blason_magyar.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 49c — AFC blason_zone × phase
# FactoMineR::CA ou ca::ca selon disponibilité
# ---------------------------------------------------------------------------

lbl_roi <- c(
  haut_droite   = "Haut droite",
  bas_droite    = "Bas droite",
  haut_gauche   = "Haut gauche",
  bas_gauche    = "Bas gauche",
  image_entiere = "Image entière (photo)"
)

ct_afc <- df_media |>
  filter(blason_present, !is.na(blason_zone), !is.na(phase_lbl)) |>
  mutate(
    roi_lbl = lbl_roi[blason_zone],
    roi_lbl = if_else(is.na(roi_lbl), blason_zone, roi_lbl)
  ) |>
  with(table(roi_lbl, phase_lbl))

# Supprimer lignes vides
ct_afc <- ct_afc[rowSums(ct_afc) > 0, , drop = FALSE]

message("\n=== 49c — Tableau AFC roi × phase ===")
print(ct_afc)

afc_done <- FALSE
if (requireNamespace("FactoMineR", quietly = TRUE) &&
    requireNamespace("factoextra",  quietly = TRUE)) {

  library(FactoMineR)
  library(factoextra)

  if (nrow(ct_afc) >= 2 && ncol(ct_afc) >= 2) {
    afc <- CA(ct_afc, graph = FALSE)
    pct1 <- round(afc$eig[1, 2], 1)
    pct2 <- round(afc$eig[2, 2], 1)

    message(sprintf("  Dim1 = %.1f%%, Dim2 = %.1f%%", pct1, pct2))

    p49c <- fviz_ca_biplot(afc,
      repel    = TRUE,
      col.row  = PAL_PHASE["2_Semi-pro"],   # positions ROI
      col.col  = PAL_PHASE["1_Artisanal"],  # phases
      title    = "AFC — position du blason (ROI) × phase"
    ) +
      labs(
        subtitle = sprintf(
          "Dim1 = %.1f%% | Dim2 = %.1f%% de l'inertie totale — %d médias avec blason",
          pct1, pct2, sum(ct_afc)
        ),
        caption = paste0(
          "Source : messages_clean.jsonl. ",
          "Points carrés = positions ROI ; points ronds = phases. ",
          "Proximité = co-occurrence."
        )
      ) +
      theme_madyar()

    save_plot(p49c, file.path(OUT, "49c_afc_roi_phase.png"),
              format = "square", width = 8, dpi = 600)
    afc_done <- TRUE
  }
} else if (requireNamespace("ca", quietly = TRUE)) {
  # Fallback : package ca (plus léger, même résultat)
  library(ca)

  if (nrow(ct_afc) >= 2 && ncol(ct_afc) >= 2) {
    afc <- ca(ct_afc)
    pct <- round(afc$sv^2 / sum(afc$sv^2) * 100, 1)

    # Extraction coordonnées pour ggplot manuel
    rc <- as.data.frame(afc$rowcoord[, 1:2])
    names(rc) <- c("Dim1", "Dim2")
    rc$label <- rownames(afc$rowcoord)
    rc$type  <- "ROI"

    cc <- as.data.frame(afc$colcoord[, 1:2])
    names(cc) <- c("Dim1", "Dim2")
    cc$label <- rownames(afc$colcoord)
    cc$type  <- "Phase"

    pts <- bind_rows(rc, cc)

    p49c <- ggplot(pts, aes(x = Dim1, y = Dim2, colour = type, shape = type)) +
      geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
      geom_vline(xintercept = 0, colour = "grey80", linewidth = 0.3) +
      geom_point(size = 3) +
      ggrepel::geom_text_repel(aes(label = label), size = 3.2, show.legend = FALSE) +
      scale_colour_manual(values = c(ROI = PAL_PHASE["2_Semi-pro"],
                                     Phase = PAL_PHASE["1_Artisanal"]),
                          name = NULL) +
      scale_shape_manual(values = c(ROI = 15, Phase = 19), name = NULL) +
      labs(
        title    = "AFC — position du blason (ROI) × phase",
        subtitle = sprintf("Dim1 = %.1f%% | Dim2 = %.1f%% — %d médias avec blason",
                           pct[1], pct[2], sum(ct_afc)),
        x = sprintf("Dim 1 (%.1f%%)", pct[1]),
        y = sprintf("Dim 2 (%.1f%%)", pct[2]),
        caption  = paste0("Source : messages_clean.jsonl. Package ca.")
      ) +
      theme(legend.position = "bottom")

    save_plot(p49c, file.path(OUT, "49c_afc_roi_phase.png"),
              format = "square", width = 8, dpi = 600)
    afc_done <- TRUE
  }
}

if (!afc_done) {
  message("49c — skippé : installer FactoMineR+factoextra ou ca.")
  message("  install.packages(c('FactoMineR', 'factoextra'))")
}

# ---------------------------------------------------------------------------
# 49d — Heatmap spatiale ROI
# Schéma "frame" : fréquence des détections par quadrant, facetté par phase
# Source : CSV per-frame (≈100K lignes) pour maximiser la granularité
# ---------------------------------------------------------------------------

csv_path <- file.path(BASE, "..", "2d_vision", "blasons", "results",
                      "blason_detection.csv")

if (!file.exists(csv_path)) {
  message("49d — CSV per-frame introuvable : ", csv_path)
} else {
  # Coordonnées des ROI dans le repère ggplot (y vers le haut = image inversée)
  roi_coords <- tibble::tribble(
    ~blason_zone,     ~xmin, ~xmax, ~ymin, ~ymax,  ~cx,   ~cy,
    "haut_droite",   0.70,  1.00,  0.70,  1.00,  0.85,  0.85,
    "bas_droite",    0.70,  1.00,  0.00,  0.30,  0.85,  0.15,
    "haut_gauche",   0.00,  0.30,  0.70,  1.00,  0.15,  0.85,
    "bas_gauche",    0.00,  0.30,  0.00,  0.30,  0.15,  0.15,
    "image_entiere", 0.35,  0.65,  0.35,  0.65,  0.50,  0.50
  )

  # Charger CSV et joindre phase
  df_csv <- read.csv(csv_path, stringsAsFactors = FALSE) |>
    filter(blason_present == "True", nzchar(blason_zone)) |>
    mutate(message_id = as.integer(message_id)) |>
    left_join(
      df_clean |> select(message_id, phase) |>
        mutate(message_id = as.integer(message_id)),
      by = "message_id"
    ) |>
    filter(!is.na(phase)) |>
    mutate(phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = phase_lbl_levels))

  message("\n=== 49d — Frames détectées par phase ===")
  print(table(df_csv$phase_lbl))

  # Compter par ROI × phase, puis normaliser dans chaque phase (profil colonne)
  df_hm <- df_csv |>
    count(phase_lbl, blason_zone, name = "n_frames") |>
    group_by(phase_lbl) |>
    mutate(pct = n_frames / sum(n_frames) * 100) |>
    ungroup() |>
    left_join(roi_coords, by = "blason_zone") |>
    filter(!is.na(xmin))  # ROI inconnue éventuelle

  # Encadré "écran" complet (fond neutre)
  frame_box <- data.frame(
    phase_lbl = factor(phase_lbl_levels, levels = phase_lbl_levels),
    xmin = 0, xmax = 1, ymin = 0, ymax = 1
  )

  # Label de n total dans la phase (pour le sous-titre de facette)
  n_par_phase <- df_csv |>
    count(phase_lbl, name = "n_total") |>
    mutate(facette = sprintf("%s\n(n=%d frames)", LBL_PHASE_SHORT[as.character(phase_lbl)], n_total))

  df_hm <- df_hm |>
    left_join(n_par_phase |> select(phase_lbl, facette), by = "phase_lbl")

  frame_box <- frame_box |>
    left_join(n_par_phase |> select(phase_lbl, facette), by = "phase_lbl")

  p49d <- ggplot(df_hm, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                             fill = pct)) +
    # Fond neutre représentant l'écran
    geom_rect(data = frame_box,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = "grey95", colour = "grey60", linewidth = 0.5, inherit.aes = FALSE) +
    # Rectangles colorés par fréquence
    geom_rect(colour = "white", linewidth = 0.6) +
    # Valeurs en %
    geom_text(
      aes(x = cx, y = cy,
          label = sprintf("%.0f%%", pct),
          colour = pct > 40),
      size = 4, fontface = "bold"
    ) +
    scale_fill_gradient(
      low  = "#EFF3FF",
      high = "#08519C",
      name = "% des\nframes détectées",
      labels = scales::label_number(suffix = " %", accuracy = 1)
    ) +
    scale_colour_manual(values = c("FALSE" = "grey15", "TRUE" = "white"), guide = "none") +
    facet_wrap(~ facette, ncol = 3) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title    = "Position du blason dans l'image par phase",
      subtitle = "Fréquence relative des détections par quadrant (frames positives SIFT uniquement)",
      x = NULL, y = NULL,
      caption  = paste0(
        "Source : blason_detection.csv — ", nrow(df_csv), " frames avec blason détecté. ",
        "Chaque cadre représente schématiquement l'image (haut = haut)."
      )
    ) +
    theme(
      axis.text  = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      legend.position = "right"
    )

  save_plot(p49d, file.path(OUT, "49d_heatmap_roi_spatial.png"),
            format = "wide_16_9", width = 12, dpi = 600)
}

# ---------------------------------------------------------------------------
# Résumé console
# ---------------------------------------------------------------------------

message("\n=== Résumé ===")
message(sprintf("  49a chi2 phase   : chi2(%d)=%.1f, p=%s",
  chi_phase$parameter, chi_phase$statistic,
  formatC(chi_phase$p.value, format = "e", digits = 2)))
message(sprintf("  49b chi2 magyar  : chi2(%d)=%.1f, p=%s, OR=%.2f [%.2f–%.2f]",
  chi_mag$parameter, chi_mag$statistic,
  formatC(chi_mag$p.value, format = "e", digits = 2),
  or_val, ci_or[1], ci_or[2]))
message("=== Terminé : exports dans ", OUT, " ===")
