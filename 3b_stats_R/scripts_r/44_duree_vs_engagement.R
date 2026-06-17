# 44_duree_vs_engagement.R — Relation durée × vues par phase
# Produit : 4_data_et_viz/44a_duree_vs_views.png, 44b_duree_vs_views_facet.png,
#           44c_correlation_duree_views.png, 44d_duree_vs_reactions.png
# Rscript 3b_stats_R/scripts_r/44_duree_vs_engagement.R

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
message("=== 44_duree_vs_engagement.R — ", nrow(df_clean), " messages chargés ===")

# Vidéos uniquement, avec durée et vues > 0
df_vid <- df_clean |>
  filter(
    media_type == "video",
    !is.na(duree_sec), duree_sec > 0,
    vues > 0
  ) |>
  mutate(
    duree_min = duree_sec / 60,
    log_views = log10(vues),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  ) |>
  filter(!is.na(phase_lbl))

n_videos <- nrow(df_vid)
message("  Vidéos avec durée + vues : ", n_videos)

# ---------------------------------------------------------------------------
# 44a — Scatter durée × vues (log) par phase + loess
# ---------------------------------------------------------------------------

plot_duree_views <- function(df, titre = "Durée vs Vues par phase") {
  ggplot(df, aes(x = duree_min, y = vues, colour = phase_lbl)) +
    geom_point(alpha = 0.35, size = 1.5) +
    geom_smooth(method = "loess", se = TRUE, linewidth = 1, alpha = 0.15,
                span = 0.7, na.rm = TRUE) +
    scale_colour_phase(short = TRUE) +
    scale_x_continuous(
      labels = label_number(suffix = " min", accuracy = 0.1),
      limits = c(0, quantile(df$duree_min, 0.98))
    ) +
    scale_y_log10(
      labels = label_number(scale_cut = cut_short_scale())
    ) +
    labs(
      title    = titre,
      subtitle = "Chaque point = 1 vidéo — courbe LOESS par phase",
      x = "Durée (minutes)", y = "Vues (échelle log)",
      caption  = paste0(n_videos, " vidéos. Axe Y log10. Tronqué au 98e percentile en X.")
    )
}

save_plot(
  plot_duree_views(df_vid),
  file.path(OUT, "44a_duree_vs_views.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 44b — Même chose, facettée par phase
# ---------------------------------------------------------------------------

plot_duree_views_facet <- function(df) {
  ggplot(df, aes(x = duree_min, y = vues)) +
    geom_point(aes(colour = phase_lbl), alpha = 0.35, size = 1.5) +
    geom_smooth(method = "loess", se = TRUE, colour = "grey30",
                linewidth = 0.8, alpha = 0.15, span = 0.7, na.rm = TRUE) +
    facet_wrap(~ phase_lbl, labeller = labeller(phase_lbl = LBL_PHASE_SHORT),
               scales = "free_x") +
    scale_colour_phase(short = TRUE) +
    scale_x_continuous(labels = label_number(suffix = " min", accuracy = 0.1)) +
    scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
    labs(
      title    = "Durée vs Vues — par phase",
      subtitle = "LOESS par phase — échelles X libres pour voir la distribution",
      x = "Durée (minutes)", y = "Vues (échelle log)",
      caption  = paste0("Source : messages_clean.jsonl — ", n_videos, " vidéos.")
    ) +
    guides(colour = "none") +
    theme_madyar_facet()
}

save_plot(
  plot_duree_views_facet(df_vid),
  file.path(OUT, "44b_duree_vs_views_facet.png"),
  format = "wide_16_9", width = 12, dpi = 600
)

# ---------------------------------------------------------------------------
# 44c — Corrélation durée × vues par phase (Spearman)
# ---------------------------------------------------------------------------

cor_phase <- df_vid |>
  group_by(phase_lbl) |>
  summarise(
    rho = cor(duree_sec, vues, method = "spearman", use = "complete.obs"),
    p_value = cor.test(duree_sec, vues, method = "spearman")$p.value,
    n = n(),
    .groups = "drop"
  ) |>
  mutate(
    signif = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE            ~ "ns"
    )
  )

plot_cor_phase <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = rho, fill = phase_lbl)) +
    geom_col(width = 0.55, alpha = 0.85) +
    geom_text(aes(label = sprintf("rho = %.3f %s\n(n=%d)", rho, signif, n)),
              vjust = -0.3, size = 3.5, fontface = "bold") +
    geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.3) +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      limits = c(min(df$rho, -0.1) * 1.3, max(df$rho, 0.1) * 1.6),
      expand = expansion(mult = c(0.05, 0.15))
    ) +
    labs(
      title    = "Corrélation durée × vues par phase",
      subtitle = "Spearman rho — négatif = vidéos courtes = plus vues",
      x = NULL, y = "Coefficient de Spearman (rho)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", sum(df$n),
        " vidéos. *** p<0.001, ** p<0.01, * p<0.05, ns = non significatif."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_cor_phase(cor_phase),
  file.path(OUT, "44c_correlation_duree_views.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 44d — Durée × réactions (même logique)
# ---------------------------------------------------------------------------

plot_duree_reactions <- function(df, titre = "Durée vs Réactions par phase") {
  df_f <- df |> filter(reactions > 0)
  ggplot(df_f, aes(x = duree_min, y = reactions, colour = phase_lbl)) +
    geom_point(alpha = 0.35, size = 1.5) +
    geom_smooth(method = "loess", se = TRUE, linewidth = 1, alpha = 0.15,
                span = 0.7, na.rm = TRUE) +
    scale_colour_phase(short = TRUE) +
    scale_x_continuous(
      labels = label_number(suffix = " min", accuracy = 0.1),
      limits = c(0, quantile(df_f$duree_min, 0.98))
    ) +
    scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
    labs(
      title    = titre,
      subtitle = "Chaque point = 1 vidéo — LOESS par phase",
      x = "Durée (minutes)", y = "Réactions (échelle log)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", nrow(df_f),
        " vidéos avec réactions > 0."
      )
    )
}

save_plot(
  plot_duree_reactions(df_vid),
  file.path(OUT, "44d_duree_vs_reactions.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Résumé
message("\n=== Corrélation durée × vues par phase ===")
print(as.data.frame(cor_phase))
message("=== Terminé : exports dans ", OUT, " ===")
