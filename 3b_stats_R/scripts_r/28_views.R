# 28_views.R — Vues Telegram : séries temporelles P1/P2/P3 + P3 stratifiée par type de média
# Produit :
#   4_data_et_viz/28_views_mois_p1_total.png       (série mensuelle P1)
#   4_data_et_viz/28b_views_mois_p1_mean.png
#   4_data_et_viz/29_views_semaine_p1_total.png
#   4_data_et_viz/29b_views_semaine_p1_mean.png
#   + variantes _cpt (changepoints Phase 1)
#   4_data_et_viz/28_views_mois_all_total.png      (toutes phases, lignes P)
#   4_data_et_viz/28_views_mois_all_mean.png
#   4_data_et_viz/28d_views_p3_par_type.png        (P3 stratifiée — filtre cohérent)
#   4_data_et_viz/28e_views_median_phase.png       (médiane globale P1/P2/P3)
# Rscript 3b_stats_R/scripts_r/28_views.R

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
message("=== 28_views.R — ", nrow(df_clean), " messages charg\u00e9s ===")

# ---------------------------------------------------------------------------
# FONCTIONS SÉRIES TEMPORELLES
# ---------------------------------------------------------------------------

plot_views_par_mois <- function(data, titre = "Vues par mois",
                                metric = c("total", "mean"),
                                phase_lines = FALSE,
                                bar_fill = unname(PAL_PHASE["1_Artisanal"])) {
  metric <- match.arg(metric)

  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(mois = as.Date(floor_date(date, "month"))) %>%
    group_by(mois) %>%
    summarise(
      vues_totales  = sum(vues, na.rm = TRUE),
      n_posts       = n(),
      vues_moy_post = vues_totales / n_posts,
      .groups = "drop"
    ) %>%
    complete(mois = seq(min(mois), max(mois), by = "month"),
             fill = list(vues_totales = 0, n_posts = 0, vues_moy_post = NA_real_)) %>%
    arrange(mois)

  y_var    <- if (metric == "total") "vues_totales" else "vues_moy_post"
  y_label  <- if (metric == "total") "Vues totales" else "Vues moyennes / post"
  subtitle <- if (metric == "total")
    "Somme des vues Telegram par mois (snapshot au moment du scraping)"
  else
    "Moyenne des vues par post publi\u00e9 ce mois"

  p <- ggplot(df, aes(x = mois, y = .data[[y_var]])) +
    geom_col(fill = bar_fill, width = 25, alpha = 0.85, na.rm = TRUE) +
    scale_x_mois(breaks = "2 months") +
    scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()),
                       expand = expansion(mult = c(0, 0.08))) +
    labs(title = titre, subtitle = subtitle, x = NULL, y = y_label,
         caption = "Source : messages_clean.jsonl. vues = compteur cumulatif Telegram.")

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

plot_views_par_semaine <- function(data, titre = "Vues par semaine",
                                   metric = c("total", "mean"),
                                   phase_lines = FALSE,
                                   fill_color = unname(PAL_PHASE["1_Artisanal"])) {
  metric <- match.arg(metric)

  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(semaine = as.Date(floor_date(date, "week", week_start = 1))) %>%
    group_by(semaine) %>%
    summarise(
      vues_totales  = sum(vues, na.rm = TRUE),
      n_posts       = n(),
      vues_moy_post = vues_totales / n_posts,
      .groups = "drop"
    ) %>%
    complete(semaine = seq(min(semaine), max(semaine), by = "week"),
             fill = list(vues_totales = 0, n_posts = 0, vues_moy_post = NA_real_)) %>%
    arrange(semaine)

  y_var    <- if (metric == "total") "vues_totales" else "vues_moy_post"
  y_label  <- if (metric == "total") "Vues totales" else "Vues moyennes / post"
  subtitle <- if (metric == "total")
    "Somme des vues Telegram par semaine (snapshot au moment du scraping)"
  else
    "Moyenne des vues par post publi\u00e9 dans la semaine"

  p <- ggplot(df, aes(x = semaine, y = .data[[y_var]])) +
    geom_area(fill = fill_color, alpha = 0.20, na.rm = TRUE) +
    geom_line(colour = fill_color, linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 1.3, colour = fill_color, na.rm = TRUE) +
    scale_x_mois(breaks = "1 month") +
    scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()),
                       expand = expansion(mult = c(0, 0.08))) +
    labs(title = titre, subtitle = subtitle, x = NULL, y = y_label,
         caption = "Source : messages_clean.jsonl. Semaines ISO (lundi = d\u00e9but).")

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# SECTION 1 — Phase 1 : séries temporelles
# ---------------------------------------------------------------------------

save_plot(
  plot_views_par_mois(df_phase(1),
    titre = "Vues par mois \u2014 Phase 1 (sept. 2022 \u2013 d\u00e9c. 2023)",
    metric = "total"),
  file.path(OUT, "28_views_mois_p1_total.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_views_par_mois(df_phase(1),
    titre = "Vues moyennes / post par mois \u2014 Phase 1",
    metric = "mean"),
  file.path(OUT, "28b_views_mois_p1_mean.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_views_par_semaine(df_phase(1),
    titre = "Vues par semaine \u2014 Phase 1 (sept. 2022 \u2013 d\u00e9c. 2023)",
    metric = "total"),
  file.path(OUT, "29_views_semaine_p1_total.png"),
  format = "wide_16_9", width = 12, dpi = 600
)
save_plot(
  plot_views_par_semaine(df_phase(1),
    titre = "Vues moyennes / post par semaine \u2014 Phase 1",
    metric = "mean"),
  file.path(OUT, "29b_views_semaine_p1_mean.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# Variantes avec changepoints
df_vm_total <- df_phase(1) %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  group_by(mois) %>%
  summarise(vues_totales = sum(vues, na.rm = TRUE), n_posts = n(), .groups = "drop") %>%
  complete(mois = seq(min(mois), max(mois), by = "month"),
           fill = list(vues_totales = 0, n_posts = 0))

df_vm_mean <- df_vm_total %>%
  mutate(vues_moy = if_else(n_posts > 0, vues_totales / n_posts, NA_real_))

df_vs_total <- df_phase(1) %>%
  filter(!is.na(date)) %>%
  mutate(semaine = as.Date(floor_date(date, "week", week_start = 1))) %>%
  group_by(semaine) %>%
  summarise(vues_totales = sum(vues, na.rm = TRUE), n_posts = n(), .groups = "drop") %>%
  complete(semaine = seq(min(semaine), max(semaine), by = "week"),
           fill = list(vues_totales = 0, n_posts = 0))

df_vs_mean <- df_vs_total %>%
  mutate(vues_moy = if_else(n_posts > 0, vues_totales / n_posts, NA_real_))

save_plot(
  add_cpt_lines(
    plot_views_par_mois(df_phase(1), titre = "Vues par mois \u2014 Phase 1 \u2014 ruptures", metric = "total"),
    compute_cpts(df_vm_total$vues_totales, df_vm_total$mois)
  ),
  file.path(OUT, "28_views_mois_p1_total_cpt.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  add_cpt_lines(
    plot_views_par_mois(df_phase(1), titre = "Vues moy/post par mois \u2014 Phase 1 \u2014 ruptures", metric = "mean"),
    compute_cpts(df_vm_mean$vues_moy, df_vm_mean$mois)
  ),
  file.path(OUT, "28b_views_mois_p1_mean_cpt.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  add_cpt_lines(
    plot_views_par_semaine(df_phase(1), titre = "Vues par semaine \u2014 Phase 1 \u2014 ruptures", metric = "total"),
    compute_cpts(df_vs_total$vues_totales, df_vs_total$semaine)
  ),
  file.path(OUT, "29_views_semaine_p1_total_cpt.png"),
  format = "wide_16_9", width = 12, dpi = 600
)
save_plot(
  add_cpt_lines(
    plot_views_par_semaine(df_phase(1), titre = "Vues moy/post par semaine \u2014 Phase 1 \u2014 ruptures", metric = "mean"),
    compute_cpts(df_vs_mean$vues_moy, df_vs_mean$semaine)
  ),
  file.path(OUT, "29b_views_semaine_p1_mean_cpt.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# ---------------------------------------------------------------------------
# SECTION 2 — Toutes phases : série mensuelle avec lignes de phase
# ---------------------------------------------------------------------------

save_plot(
  plot_views_par_mois(df_clean,
    titre = "Vues par mois \u2014 corpus complet (P1\u2013P3)",
    metric = "total",
    phase_lines = TRUE,
    bar_fill = "grey40"),
  file.path(OUT, "28_views_mois_all_total.png"),
  format = "wide_16_9", width = 12, dpi = 600
)
save_plot(
  plot_views_par_mois(df_clean,
    titre = "Vues moyennes / post \u2014 corpus complet (P1\u2013P3)",
    metric = "mean",
    phase_lines = TRUE,
    bar_fill = "grey40"),
  file.path(OUT, "28_views_mois_all_mean.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# ---------------------------------------------------------------------------
# SECTION 3 — P3 : vues par type de média (filtre cohérent)
# ---------------------------------------------------------------------------
# Filtre !vues_is_missing appliqué identiquement au global et aux strates,
# sinon la médiane globale (texte_uniquement imputés 0) diverge des strates.

df_p3_v <- df_phase(3) %>%
  filter(!vues_is_missing) %>%
  mutate(
    type_lbl = case_when(
      media_type == "video"            ~ "Vid\u00e9o",
      media_type == "photo"            ~ "Photo",
      media_type == "texte_uniquement" ~ "Texte seul",
      TRUE                             ~ "Autre"
    ),
    type_lbl = factor(type_lbl, levels = c("Vid\u00e9o", "Photo", "Texte seul", "Autre"))
  )

stats_p3 <- df_p3_v %>%
  group_by(type_lbl) %>%
  summarise(n = n(), mediane = median(vues), q1 = quantile(vues, 0.25),
            q3 = quantile(vues, 0.75), .groups = "drop")

# Si cette assertion échoue, le filtre a divergé entre global et strates.
stopifnot("strate \u222a global incoh\u00e9rent" = sum(stats_p3$n) == nrow(df_p3_v))

med_global_p3 <- median(df_p3_v$vues)
message("=== P3 distribution vues ===")
print(as.data.frame(stats_p3))
message(sprintf("P3 global : n=%d  m\u00e9diane=%s",
                nrow(df_p3_v),
                format(round(med_global_p3), big.mark = " ")))

p_p3_type <- ggplot(df_p3_v, aes(x = type_lbl, y = vues, fill = type_lbl)) +
  geom_violin(alpha = 0.6, colour = NA, scale = "area") +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.85, colour = "grey20") +
  geom_hline(yintercept = med_global_p3, colour = "firebrick",
             linetype = "dashed", linewidth = 0.7) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_fill_cat(guide = "none") +
  labs(
    title    = "Vues par message \u2014 P3 par type de m\u00e9dia",
    subtitle = sprintf(
      "%d messages avec compteur renseign\u00e9 \u2014 m\u00e9d. globale P3 : %s (ligne rouge)",
      nrow(df_p3_v),
      format(round(med_global_p3), big.mark = " ")
    ),
    x = NULL, y = "Vues (log\u2081\u2080)",
    caption = paste0(
      "Source : messages_clean.jsonl. Compteur cumulatif Telegram.\n",
      "Filtre : vues_is_missing = FALSE (NA imput\u00e9s 0 dans r_source exclus)."
    )
  )

save_plot(p_p3_type, file.path(OUT, "28d_views_p3_par_type.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# SECTION 4 — Médiane des vues par phase
# ---------------------------------------------------------------------------
# Même filtre !vues_is_missing pour cohérence inter-sections.

PHASE_KEY <- c("1" = "1_Artisanal", "2" = "2_Semi-pro", "3" = "3_Institutionnel")

df_med_phase <- df_clean %>%
  filter(!is.na(phase), !vues_is_missing) %>%
  group_by(phase) %>%
  summarise(n = n(), mediane = median(vues), .groups = "drop") %>%
  mutate(
    phase_key = PHASE_KEY[as.character(phase)],
    phase_lbl = factor(
      phase_key,
      levels = names(PAL_PHASE),
      labels = c("P1\nArtisanal", "P2\nSemi-pro", "P3\nInstitutionnel")
    )
  )

p_med_phase <- ggplot(df_med_phase,
                      aes(x = phase_lbl, y = mediane, fill = phase_key)) +
  geom_col(alpha = 0.85, width = 0.55) +
  geom_text(aes(label = format(round(mediane), big.mark = " ")),
            vjust = -0.5, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = PAL_PHASE, guide = "none") +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "M\u00e9diane des vues par phase \u2014 corpus complet",
    subtitle = sprintf("Messages avec compteur renseign\u00e9 (%s au total)",
                       format(sum(df_med_phase$n), big.mark = " ")),
    x = NULL, y = "Vues (m\u00e9diane)",
    caption = "Source : messages_clean.jsonl. Filtre : vues_is_missing = FALSE."
  )

save_plot(p_med_phase, file.path(OUT, "28e_views_median_phase.png"),
          format = "wide_16_9", width = 8, dpi = 600)

message("=== Termin\u00e9 : 12 exports dans ", OUT, " ===")
