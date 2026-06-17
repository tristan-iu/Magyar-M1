# 42_ocr_watermark.R — Branding visuel : watermark, logo ПТАХИ МАДЯРА, 414 ОБр
# Produit : 4_data_et_viz/42a_watermark_mois.png, 42b_brand_type_mois.png,
#           42c_brand_phase_dodge.png, 42d_brand_timeline.png
# Rscript 3b_stats_R/scripts_r/42_ocr_watermark.R

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
message("=== 42_ocr_watermark.R — ", nrow(df_clean), " messages chargés ===")

# Colonnes OCR
df_clean <- df_clean |>
  ensure_col("ocr_filigrane_present", FALSE) |>
  ensure_col("ocr_filigrane_texte", NA_character_) |>
  ensure_col("ocr_texte", NA_character_)

# Combiner les deux champs OCR pour la recherche
df_brand <- df_clean |>
  filter(!is.na(mois)) |>
  mutate(
    mois = as.Date(mois),
    ocr_all = paste0(
      replace_na(as.character(ocr_texte), ""), " ",
      replace_na(as.character(ocr_filigrane_texte), "")
    ),
    # Watermark générique (détecté par vision_batch)
    has_watermark = as.logical(ocr_filigrane_present) & !is.na(ocr_filigrane_present),
    # Logo ПТАХИ МАДЯРА (branding personnel → unité)
    has_ptahi = grepl("ПТАХ", ocr_all, ignore.case = TRUE),
    # Mention МАДЯР dans watermark (nom personnel)
    has_madyar_wm = has_watermark & grepl("МАДЯР", as.character(ocr_filigrane_texte), ignore.case = TRUE),
    # 414 ОБр / ОПУБАС (branding institutionnel brigade)
    has_414 = grepl("414", ocr_all),
    # Catégorie de branding
    brand_type = case_when(
      has_414            ~ "414 ОБр (brigade)",
      has_ptahi          ~ "Птахи Мадяра (unité)",
      has_madyar_wm      ~ "Мадяр (personnel)",
      has_watermark       ~ "Autre watermark",
      TRUE                ~ "Aucun"
    ),
    # Hiérarchie : tout branding confondu
    has_any_brand = has_watermark | has_ptahi | has_414,
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  )

# Ne garder que les médias (watermark n'a pas de sens sur texte seul)
df_media <- df_brand |> filter(has_media)
n_media <- nrow(df_media)
message("  Médias avec OCR : ", n_media)

# ---------------------------------------------------------------------------
# 42a — % de médias avec watermark par mois
# ---------------------------------------------------------------------------

df_mois_wm <- df_media |>
  group_by(mois) |>
  summarise(
    pct_watermark = mean(has_any_brand, na.rm = TRUE) * 100,
    pct_414       = mean(has_414, na.rm = TRUE) * 100,
    pct_ptahi     = mean(has_ptahi, na.rm = TRUE) * 100,
    pct_madyar    = mean(has_madyar_wm, na.rm = TRUE) * 100,
    n_media       = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

plot_wm_mois <- function(df, titre = "Présence de branding visuel par mois") {
  ggplot(df, aes(x = mois, y = pct_watermark)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_col(fill = PAL_PHASE["2_Semi-pro"], width = 25, alpha = 0.85, na.rm = TRUE) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " %", accuracy = 1),
      limits = c(0, 100),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title    = titre,
      subtitle = "% de médias avec watermark/logo détecté (OCR + regex)",
      x = NULL, y = "Médias avec branding (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_media,
        " médias. Détection : МАДЯР | ПТАХИ | 414 dans ocr_texte/watermark_text."
      )
    )
}

save_plot(
  plot_wm_mois(df_mois_wm),
  file.path(OUT, "42a_watermark_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 42b — Évolution des types de branding par mois (stacked)
# ---------------------------------------------------------------------------

# Ordonner les niveaux du plus personnel au plus institutionnel
brand_levels <- c("414 ОБр (brigade)", "Птахи Мадяра (unité)",
                  "Мадяр (personnel)", "Autre watermark", "Aucun")

df_brand_mois <- df_media |>
  mutate(brand_type = factor(brand_type, levels = brand_levels)) |>
  count(mois, brand_type) |>
  complete(
    mois = seq(min(mois), max(mois), by = "month"),
    brand_type = factor(brand_levels, levels = brand_levels),
    fill = list(n = 0)
  )

# Palette spécifique : du plus institutionnel (foncé) au plus absent (clair)
pal_brand <- c(
  "414 ОБр (brigade)"      = "#D95F02",
  "Птахи Мадяра (unité)"   = "#1B9E77",
  "Мадяр (personnel)"      = "#7570B3",
  "Autre watermark"         = "#999999",
  "Aucun"                   = "#E0E0E0"
)

plot_brand_evolution <- function(df, titre = "Type de branding par mois") {
  ggplot(df, aes(x = mois, y = n, fill = brand_type)) +
    geom_col(position = "fill", colour = "white", linewidth = 0.2, width = 25) +
    geom_phase_lines() +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_manual(values = pal_brand) +
    labs(
      title    = titre,
      subtitle = "Part relative — hiérarchie : 414 > Птахи > Мадяр > Autre > Aucun",
      x = NULL, y = "Proportion (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_media,
        " médias. Priorité au branding le plus institutionnel quand plusieurs détectés."
      )
    )
}

save_plot(
  plot_brand_evolution(df_brand_mois),
  file.path(OUT, "42b_brand_type_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 42c — % branding par phase (barplot groupé)
# ---------------------------------------------------------------------------

df_brand_phase <- df_media |>
  filter(!is.na(phase_lbl)) |>
  group_by(phase_lbl) |>
  summarise(
    pct_any    = mean(has_any_brand) * 100,
    pct_madyar = mean(has_madyar_wm) * 100,
    pct_ptahi  = mean(has_ptahi) * 100,
    pct_414    = mean(has_414) * 100,
    n          = n(),
    .groups = "drop"
  )

df_brand_long <- df_brand_phase |>
  select(phase_lbl, pct_madyar, pct_ptahi, pct_414) |>
  pivot_longer(-phase_lbl, names_to = "brand", values_to = "pct") |>
  mutate(brand = factor(recode(brand,
    pct_madyar = "Мадяр (personnel)",
    pct_ptahi  = "Птахи Мадяра (unité)",
    pct_414    = "414 ОБр (brigade)"
  ), levels = c("Мадяр (personnel)", "Птахи Мадяра (unité)", "414 ОБр (brigade)")))

plot_brand_phase <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = pct, fill = brand)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    scale_fill_manual(values = pal_brand) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      labels = label_number(suffix = " %", accuracy = 1),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title    = "Branding visuel par phase",
      subtitle = paste0(
        "% de médias contenant chaque type de marque — ",
        sum(df_brand_phase$n), " médias"
      ),
      x = NULL, y = "Médias avec branding (%)",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(df_brand_phase$phase_lbl)],
              df_brand_phase$n), collapse = " | "), "."
      )
    )
}

save_plot(
  plot_brand_phase(df_brand_long),
  file.path(OUT, "42c_brand_phase_dodge.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 42d — Timeline : première apparition de chaque type de branding
# ---------------------------------------------------------------------------

first_brand <- df_media |>
  filter(has_any_brand) |>
  group_by(brand_type) |>
  summarise(
    premiere = min(jour, na.rm = TRUE),
    derniere = max(jour, na.rm = TRUE),
    n_total  = n(),
    .groups = "drop"
  ) |>
  filter(brand_type != "Aucun", brand_type != "Autre watermark") |>
  mutate(brand_type = factor(brand_type,
    levels = c("Мадяр (personnel)", "Птахи Мадяра (unité)", "414 ОБр (brigade)")))

plot_timeline_brand <- function(df) {
  ggplot(df, aes(y = brand_type, colour = brand_type)) +
    geom_segment(aes(x = premiere, xend = derniere, yend = brand_type),
                 linewidth = 3, alpha = 0.7) +
    geom_point(aes(x = premiere), size = 4) +
    geom_point(aes(x = derniere), size = 4, shape = 1) +
    geom_vline(xintercept = as.numeric(PHASE_DATES),
               colour = "#888888", linetype = "dashed", linewidth = 0.5) +
    scale_colour_manual(values = pal_brand) +
    scale_x_date(date_breaks = "3 months", date_labels = "%b\n%Y") +
    labs(
      title    = "Chronologie du branding visuel",
      subtitle = "Première et dernière apparition détectée (OCR)",
      x = NULL, y = NULL,
      caption  = paste0(
        "Source : messages_clean.jsonl. Point plein = première ; cercle = dernière. ",
        "Lignes verticales = frontières de phase."
      )
    ) +
    guides(colour = "none") +
    theme(axis.text.y = element_text(size = 12, face = "bold"))
}

save_plot(
  plot_timeline_brand(first_brand),
  file.path(OUT, "42d_brand_timeline.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Résumé
message("\n=== Branding par phase ===")
print(as.data.frame(df_brand_phase))
message("\n=== Premières apparitions ===")
print(as.data.frame(first_brand))
message("=== Terminé : exports dans ", OUT, " ===")
