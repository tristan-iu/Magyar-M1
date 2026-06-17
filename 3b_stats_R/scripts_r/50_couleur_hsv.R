# 50_couleur_hsv.R — Analyse couleur HSV des keyframes
# Produit : 4_data_et_viz/50a_entropie_mois.png, 50b_boxplots_hsv.png, 50c_pca_scatter_phase.png,
#           50d_correlations_hsv.png, 50e_pca_scree.png, 50f_cluster_phase.png
# Rscript 3b_stats_R/scripts_r/50_couleur_hsv.R

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

COULEUR_DIR <- file.path(BASE, "..", "4_data_et_viz", "couleurs")
stats_path   <- file.path(COULEUR_DIR, "couleur_stats.csv")
pca_path     <- file.path(COULEUR_DIR, "couleur_pca.csv")
var_path     <- file.path(COULEUR_DIR, "couleur_pca_variance.csv")
clusters_path <- file.path(COULEUR_DIR, "couleur_clusters.csv")

if (!file.exists(stats_path))
  stop("CSV couleur introuvable : ", stats_path,
       "\nLancer couleur_batch.py puis couleur_espace.py d'abord.")

df_hsv     <- read.csv(stats_path,   stringsAsFactors = FALSE)
df_pca     <- read.csv(pca_path,     stringsAsFactors = FALSE)
df_var     <- read.csv(var_path,     stringsAsFactors = FALSE)
df_clust   <- read.csv(clusters_path, stringsAsFactors = FALSE)

message("=== 50_couleur_hsv.R — ", nrow(df_hsv), " messages avec stats couleur ===")

# ---------------------------------------------------------------------------
# Jointure avec df_clean (source de vérité pour phase, mois, etc.)
# ---------------------------------------------------------------------------

df_col <- df_clean |>
  inner_join(df_hsv, by = "message_id") |>
  filter(!is.na(phase), !is.na(mois)) |>
  mutate(
    mois = as.Date(mois),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = names(PAL_PHASE))
  )

n_total <- nrow(df_col)
message("  Jointure OK : ", n_total, " messages phase+couleur")

# ---------------------------------------------------------------------------
# 50a — Entropie chromatique par mois
# ---------------------------------------------------------------------------

df_mois <- df_col |>
  group_by(mois) |>
  summarise(
    entropy_med = median(hsv_entropy, na.rm = TRUE),
    entropy_q1  = quantile(hsv_entropy, 0.25, na.rm = TRUE),
    entropy_q3  = quantile(hsv_entropy, 0.75, na.rm = TRUE),
    sat_med     = median(hsv_s_mean, na.rm = TRUE),
    val_med     = median(hsv_v_mean, na.rm = TRUE),
    s_inter_med = median(hsv_s_inter, na.rm = TRUE),
    n           = n(),
    .groups = "drop"
  )

plot_entropy_mois <- ggplot(df_mois, aes(x = mois, y = entropy_med)) +
  geom_phase_bands() +
  geom_phase_lines() +
  geom_ribbon(aes(ymin = entropy_q1, ymax = entropy_q3),
              fill = PAL_PHASE["1_Artisanal"], alpha = 0.15) +
  geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.8) +
  geom_point(colour = PAL_PHASE["1_Artisanal"], size = 1.6) +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(limits = c(NA, 12),
                     breaks = seq(6, 12, 1),
                     sec.axis = sec_axis(~ . / 12 * 100,
                                         name = "% du maximum théorique",
                                         labels = label_number(suffix = "%"))) +
  labs(
    title    = "Entropie chromatique HSV par mois",
    subtitle = "Médiane mensuelle (Shannon sur histogramme 16³) — max théorique = 12 bits",
    x = NULL, y = "Entropie (bits)",
    caption  = paste0(
      "Source : couleur_stats.csv — ", n_total, " messages. ",
      "Ruban = IQR. Baisse = palette plus étroite (institutional look)."
    )
  )

save_plot(plot_entropy_mois, file.path(OUT, "50a_entropie_mois.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 50b — Boxplots HSV par phase (entropie, saturation, cohérence inter-frames)
# ---------------------------------------------------------------------------

df_long <- df_col |>
  select(phase_lbl,
         `Entropie (bits)`      = hsv_entropy,
         `Saturation moy.`      = hsv_s_mean,
         `Luminosité moy.`      = hsv_v_mean,
         `Ecart-type S inter`   = hsv_s_inter) |>
  pivot_longer(-phase_lbl, names_to = "metrique", values_to = "valeur") |>
  mutate(metrique = factor(metrique, levels = c(
    "Entropie (bits)", "Saturation moy.",
    "Luminosité moy.", "Ecart-type S inter"
  )))

n_par_phase <- df_col |> count(phase_lbl)
cap_phase   <- paste(sprintf("%s : n=%d",
                             LBL_PHASE_SHORT[as.character(n_par_phase$phase_lbl)],
                             n_par_phase$n), collapse = " | ")

plot_boxplots_hsv <- ggplot(df_long, aes(x = phase_lbl, y = valeur, fill = phase_lbl)) +
  geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.4,
               width = 0.6, colour = "grey25", linewidth = 0.3) +
  facet_wrap(~ metrique, scales = "free_y", nrow = 1) +
  scale_fill_phase(short = TRUE) +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  labs(
    title    = "Distributions HSV par phase",
    subtitle = "Quatre indicateurs clés de la signature chromatique",
    x = NULL, y = NULL,
    caption  = paste0("Source : couleur_stats.csv. ", cap_phase, ".")
  ) +
  theme_madyar_facet() +
  theme(legend.position = "none")

save_plot(plot_boxplots_hsv, file.path(OUT, "50b_boxplots_hsv.png"),
          format = "wide_16_9", width = 12, dpi = 600)

# ---------------------------------------------------------------------------
# 50c — Scatter PCA coloré par phase
# ---------------------------------------------------------------------------

df_pca_j <- df_pca |>
  select(message_id, pc1, pc2) |>
  inner_join(df_clean |> select(message_id, phase), by = "message_id") |>
  filter(!is.na(phase)) |>
  mutate(phase_lbl = factor(case_when(
    phase == 1L ~ "1_Artisanal",
    phase == 2L ~ "2_Semi-pro",
    phase == 3L ~ "3_Institutionnel"
  ), levels = names(PAL_PHASE)))

var_pc1 <- df_var$variance_expliquee[df_var$composante == "PC1"] * 100
var_pc2 <- df_var$variance_expliquee[df_var$composante == "PC2"] * 100

# Zoom sur le noyau central : quelques outliers écrasent la masse principale
zoom_q <- 0.99
xlim_pc <- quantile(df_pca_j$pc1, c(1 - zoom_q, zoom_q), na.rm = TRUE)
ylim_pc <- quantile(df_pca_j$pc2, c(1 - zoom_q, zoom_q), na.rm = TRUE)

plot_pca_scatter <- ggplot(df_pca_j, aes(x = pc1, y = pc2, colour = phase_lbl)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_point(alpha = 0.55, size = 1.4, stroke = 0) +
  stat_ellipse(level = 0.8, linewidth = 0.6, linetype = "dashed") +
  scale_colour_phase(short = TRUE) +
  coord_cartesian(xlim = xlim_pc, ylim = ylim_pc) +
  labs(
    title    = "Projection PCA des histogrammes HSV",
    subtitle = sprintf("PC1 %.1f %% — PC2 %.1f %% de la variance",
                       var_pc1, var_pc2),
    x = sprintf("PC1 (%.1f %%)", var_pc1),
    y = sprintf("PC2 (%.1f %%)", var_pc2),
    caption  = paste0(
      "Source : couleur_pca.csv — ", nrow(df_pca_j), " messages. ",
      "Ellipses = 80 % des points par phase (zoom q99, outliers masqués)."
    )
  ) +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1)))

save_plot(plot_pca_scatter, file.path(OUT, "50c_pca_scatter_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 50d — Corrélations couleur × signaux corpus (Spearman)
# ---------------------------------------------------------------------------

df_corr <- df_col |>
  ensure_col("scene_coupes_par_min", NA_real_) |>
  ensure_col("visages_magyar_detections", NA_real_) |>
  ensure_col("blason_present", NA) |>
  mutate(
    scene_coupes_par_min = suppressWarnings(as.numeric(scene_coupes_par_min)),
    blason_present     = suppressWarnings(as.logical(blason_present)),
    blason_num         = ifelse(is.na(blason_present), NA_real_,
                                as.numeric(blason_present)),
    magyar_num         = suppressWarnings(as.numeric(visages_magyar_detections)),
    views_log          = log10(pmax(vues, 1))
  )

vars_hsv <- c("hsv_entropy", "hsv_s_mean", "hsv_v_mean",
              "hsv_s_inter", "hsv_v_inter")
vars_cor <- c("scene_coupes_par_min", "parole_ratio", "duree_sec",
              "magyar_num", "blason_num", "views_log")

mat_corr <- outer(vars_hsv, vars_cor, Vectorize(function(a, b) {
  x <- df_corr[[a]]; y <- df_corr[[b]]
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 30) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "spearman"))
}))
rownames(mat_corr) <- vars_hsv
colnames(mat_corr) <- vars_cor

df_corr_long <- as.data.frame.table(mat_corr, responseName = "rho") |>
  rename(var_hsv = Var1, var_corpus = Var2) |>
  mutate(
    var_hsv = factor(var_hsv, levels = rev(c(
      "hsv_entropy", "hsv_s_mean", "hsv_v_mean",
      "hsv_s_inter", "hsv_v_inter"
    )), labels = rev(c(
      "Entropie", "Saturation moy.", "Luminosité moy.",
      "E.-T. S inter", "E.-T. V inter"
    ))),
    var_corpus = factor(var_corpus, levels = vars_cor, labels = c(
      "Cuts / min", "Speech ratio", "Durée (s)",
      "Magyar détecté", "Blason présent", "Vues (log10)"
    ))
  )

plot_corr_heatmap <- ggplot(df_corr_long, aes(x = var_corpus, y = var_hsv, fill = rho)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = ifelse(is.na(rho), "", sprintf("%.2f", rho))),
            size = 3.2, colour = "grey10") +
  scale_fill_gradient2(low = "#B2182B", mid = "white",
                       high = unname(PAL_PHASE["1_Artisanal"]),
                       midpoint = 0, limits = c(-0.5, 0.5),
                       oob = scales::squish,
                       name = "ρ Spearman") +
  labs(
    title    = "Corrélations couleur × signaux corpus",
    subtitle = "Spearman sur messages avec stats HSV complètes",
    x = NULL, y = NULL,
    caption  = paste0(
      "Source : couleur_stats.csv × messages_clean.jsonl — ",
      n_total, " messages. Valeurs |ρ| > 0.20 interprétables."
    )
  ) +
  theme_madyar(x_text_angle = 25, x_text_hjust = 1) +
  theme(
    panel.grid = element_blank(),
    legend.position = "right"
  )

save_plot(plot_corr_heatmap, file.path(OUT, "50d_correlations_hsv.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 50e — Scree plot (variance expliquée par composante)
# ---------------------------------------------------------------------------

df_var_plot <- df_var |>
  mutate(
    composante = factor(composante, levels = df_var$composante),
    pct_indiv  = variance_expliquee * 100,
    pct_cumul  = cumul * 100
  )

plot_scree <- ggplot(df_var_plot, aes(x = composante)) +
  geom_col(aes(y = pct_indiv),
           fill = PAL_PHASE["1_Artisanal"], alpha = 0.85,
           colour = "grey25", linewidth = 0.3, width = 0.6) +
  geom_line(aes(y = pct_cumul, group = 1),
            colour = PAL_PHASE["2_Semi-pro"], linewidth = 1) +
  geom_point(aes(y = pct_cumul),
             colour = PAL_PHASE["2_Semi-pro"], size = 2.5) +
  geom_text(aes(y = pct_indiv, label = sprintf("%.1f %%", pct_indiv)),
            vjust = -0.6, size = 3.5) +
  geom_text(aes(y = pct_cumul, label = sprintf("cumul %.1f %%", pct_cumul)),
            vjust = -1, size = 3, colour = PAL_PHASE["2_Semi-pro"],
            fontface = "italic") +
  scale_y_continuous(limits = c(0, 100),
                     labels = label_number(suffix = " %")) +
  labs(
    title    = "PCA couleur — variance expliquée",
    subtitle = "Barres = composante individuelle ; courbe = cumul",
    x = NULL, y = "Variance expliquée",
    caption  = "Source : couleur_pca_variance.csv"
  )

save_plot(plot_scree, file.path(OUT, "50e_pca_scree.png"),
          format = "wide_16_9", width = 9, dpi = 600)

# ---------------------------------------------------------------------------
# 50f — Table contingence cluster × phase
# ---------------------------------------------------------------------------

df_cluster_j <- df_clust |>
  select(message_id, cluster) |>
  inner_join(df_clean |> select(message_id, phase), by = "message_id") |>
  filter(!is.na(phase)) |>
  mutate(
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = names(PAL_PHASE)),
    cluster_lbl = factor(paste0("C", cluster))
  )

df_cluster_pct <- df_cluster_j |>
  count(phase_lbl, cluster_lbl) |>
  group_by(phase_lbl) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

plot_cluster_phase <- ggplot(df_cluster_pct,
                             aes(x = phase_lbl, y = pct, fill = cluster_lbl)) +
  geom_col(position = "stack", colour = "white", linewidth = 0.3, width = 0.6) +
  geom_text(aes(label = ifelse(pct > 5, sprintf("%.0f %%", pct), "")),
            position = position_stack(vjust = 0.5),
            size = 3.2, colour = "white", fontface = "bold") +
  scale_fill_cat() +
  scale_x_discrete(labels = LBL_PHASE_SHORT) +
  scale_y_continuous(labels = label_number(suffix = " %"),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Clusters chromatiques K-means × phase",
    subtitle = "Répartition des clusters d'histogrammes HSV au sein de chaque phase",
    x = NULL, y = "Part de chaque cluster",
    caption  = paste0(
      "Source : couleur_clusters.csv — ", nrow(df_cluster_j), " messages. ",
      "Un cluster dominant par phase = signature chromatique distincte."
    )
  )

save_plot(plot_cluster_phase, file.path(OUT, "50f_cluster_phase.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# Résumés console
# ---------------------------------------------------------------------------

message("\n=== Médianes HSV par phase ===")
df_summary <- df_col |>
  group_by(phase_lbl) |>
  summarise(
    n          = n(),
    entropie   = median(hsv_entropy, na.rm = TRUE),
    saturation = median(hsv_s_mean, na.rm = TRUE),
    luminosite = median(hsv_v_mean, na.rm = TRUE),
    s_inter    = median(hsv_s_inter, na.rm = TRUE),
    .groups = "drop"
  )
print(as.data.frame(df_summary))

message("\n=== Corrélations HSV × corpus (Spearman) ===")
print(round(mat_corr, 3))

message("\n=== Contingence cluster × phase ===")
print(with(df_cluster_j, table(phase_lbl, cluster_lbl)))

message("=== Terminé : 6 figures dans ", OUT, " ===")
