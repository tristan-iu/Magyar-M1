# 33_afc_lexicale.R — AFC termes × phases (FactoMineR)
# Produit : 4_data_et_viz/33a_afc_lexicale_biplot.png, 33b-c_afc_contrib_dim{1,2}.png,
#           33d_afc_eboulis.png + 3 CSV (coordonnées, valeurs propres)
# Rscript 3b_stats_R/scripts_r/33_afc_lexicale.R
# Requiert : install.packages(c("FactoMineR", "factoextra", "ggrepel"))

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

if (!requireNamespace("FactoMineR", quietly = TRUE))
  stop("install.packages('FactoMineR')")
if (!requireNamespace("factoextra", quietly = TRUE))
  stop("install.packages('factoextra')")
if (!requireNamespace("ggrepel", quietly = TRUE))
  stop("install.packages('ggrepel')")

library(FactoMineR)
library(factoextra)
library(ggrepel)
library(readr)
library(dplyr)
library(tidyr)

LEMMES_CSV <- file.path(dirname(BASE), "4_data_et_viz", "lexico", "lemmes_combined.csv")
message("=== 33_afc_lexicale.R — chargement ", LEMMES_CSV, " ===")

# ---------------------------------------------------------------------------
# 1. CHARGEMENT & FILTRAGE
# ---------------------------------------------------------------------------

df_lemmes <- read_csv(LEMMES_CSV, show_col_types = FALSE)
message(nrow(df_lemmes), " tokens chargés, ", n_distinct(df_lemmes$lemma), " lemmes distincts")

POS_RETENUS <- c("NOUN", "VERB", "ADJ", "PRON", "ADV")
PHASES <- c("1_Artisanal", "2_Semi-pro", "3_Institutionnel")

df_filtre <- df_lemmes %>%
  filter(pos %in% POS_RETENUS, !is.na(lemma), lemma != "") %>%
  # Normalise "P1_Artisanal" → "1_Artisanal" (convention R vs Python)
  mutate(phase = sub("^P(\\d)", "\\1", phase),
         phase = factor(phase, levels = PHASES))

message(nrow(df_filtre), " tokens après filtre POS")

# ---------------------------------------------------------------------------
# 2. MATRICE TERMES × PHASES
# ---------------------------------------------------------------------------

mat_long <- df_filtre %>%
  count(lemma, phase, name = "n") %>%
  pivot_wider(names_from = phase, values_from = n, values_fill = 0)

mat_long <- mat_long %>%
  mutate(
    total    = rowSums(across(all_of(PHASES))),
    n_phases = rowSums(across(all_of(PHASES)) > 0)
  ) %>%
  filter(total >= 30, n_phases >= 2) %>%
  select(-total, -n_phases)

message(nrow(mat_long), " lemmes retenus (≥30 occ., ≥2 phases)")

mat <- as.data.frame(mat_long)
rownames(mat) <- mat$lemma
mat <- mat[, PHASES]

# ---------------------------------------------------------------------------
# 3. AFC
# ---------------------------------------------------------------------------

res_ca <- CA(mat, graph = FALSE)

eig <- res_ca$eig

# ===== OUTPUT CONCRET =====
message("\n╔══════════════════════════════════════════════════════╗")
message("║            OUTPUT AFC — VALEURS PROPRES              ║")
message("╚══════════════════════════════════════════════════════╝")
message(sprintf("\n%-5s %-15s %-20s %-20s", "Dim", "Valeur propre", "% variance", "% cumulé"))
message(strrep("─", 62))
for (i in seq_len(nrow(eig))) {
  message(sprintf("%-5d %-15.6f %-20.2f %-20.2f",
                  i, eig[i, 1], eig[i, 2], eig[i, 3]))
}

message("\n╔══════════════════════════════════════════════════════╗")
message("║        COORDONNÉES DES COLONNES (PHASES)             ║")
message("╚══════════════════════════════════════════════════════╝")
col_coord <- as.data.frame(res_ca$col$coord)
col_cos2  <- as.data.frame(res_ca$col$cos2)
col_contrib <- as.data.frame(res_ca$col$contrib)
message(sprintf("\n%-22s %-10s %-10s %-10s %-10s %-10s %-10s",
                "Phase", "Dim1", "Dim2", "cos2_1", "cos2_2", "ctr_1", "ctr_2"))
message(strrep("─", 72))
for (i in seq_len(nrow(col_coord))) {
  message(sprintf("%-22s %-10.4f %-10.4f %-10.4f %-10.4f %-10.2f %-10.2f",
                  rownames(col_coord)[i],
                  col_coord[i, 1], col_coord[i, 2],
                  col_cos2[i, 1], col_cos2[i, 2],
                  col_contrib[i, 1], col_contrib[i, 2]))
}

# Top contributions lignes (termes)
contrib_dim1 <- res_ca$row$contrib[, 1]
contrib_dim2 <- res_ca$row$contrib[, 2]
contrib_total <- contrib_dim1 + contrib_dim2

N_TOP <- 60
top_lemmes <- names(sort(contrib_total, decreasing = TRUE)[1:min(N_TOP, length(contrib_total))])

message("\n╔══════════════════════════════════════════════════════╗")
message("║      TOP 20 LEMMES — CONTRIBUTION DIM1               ║")
message("╚══════════════════════════════════════════════════════╝")
top20_d1 <- sort(contrib_dim1, decreasing = TRUE)[1:20]
message(sprintf("\n%-20s %-12s %-10s", "Lemme", "Contrib_Dim1", "Coord_Dim1"))
message(strrep("─", 44))
for (nm in names(top20_d1)) {
  message(sprintf("%-20s %-12.3f %-10.4f",
                  nm, top20_d1[nm], res_ca$row$coord[nm, 1]))
}

message("\n╔══════════════════════════════════════════════════════╗")
message("║      TOP 20 LEMMES — CONTRIBUTION DIM2               ║")
message("╚══════════════════════════════════════════════════════╝")
top20_d2 <- sort(contrib_dim2, decreasing = TRUE)[1:20]
message(sprintf("\n%-20s %-12s %-10s", "Lemme", "Contrib_Dim2", "Coord_Dim2"))
message(strrep("─", 44))
for (nm in names(top20_d2)) {
  message(sprintf("%-20s %-12.3f %-10.4f",
                  nm, top20_d2[nm], res_ca$row$coord[nm, 2]))
}
message("")

# ---------------------------------------------------------------------------
# 4. GRAPHIQUE — ÉBOULIS DES VALEURS PROPRES (scree plot)
# ---------------------------------------------------------------------------

eig_df <- data.frame(
  dimension = seq_len(nrow(eig)),
  valeur_propre = eig[, 1],
  pct_variance  = eig[, 2],
  pct_cumul     = eig[, 3]
)

# Seuil Kaiser (1 / nrow(eig))
kaiser_threshold <- 100 / nrow(eig)

p_scree <- ggplot(eig_df, aes(x = dimension)) +
  # Barres % variance
  geom_col(aes(y = pct_variance),
           fill = PAL_PHASE["2_Semi-pro"], alpha = 0.85, width = 0.6) +
  # Courbe % cumulé
  geom_line(aes(y = pct_cumul), colour = PAL_PHASE["1_Artisanal"],
            linewidth = 0.9, linetype = "solid") +
  geom_point(aes(y = pct_cumul), colour = PAL_PHASE["1_Artisanal"],
             size = 3, shape = 16) +
  # Seuil Kaiser
  geom_hline(yintercept = kaiser_threshold,
             colour = "firebrick", linetype = "dashed", linewidth = 0.5) +
  annotate("text", x = nrow(eig_df) - 0.3, y = kaiser_threshold + 2.5,
           label = sprintf("Seuil Kaiser (%.1f%%)", kaiser_threshold),
           colour = "firebrick", size = 3.2, hjust = 1) +
  # Labels % sur barres
  geom_text(aes(y = pct_variance + 1.5,
                label = sprintf("%.1f%%", pct_variance)),
            size = 3.2, colour = "grey20", fontface = "bold") +
  scale_x_continuous(breaks = eig_df$dimension,
                     labels = paste0("Dim ", eig_df$dimension)) +
  scale_y_continuous(
    name = "% de variance expliquée (barres)",
    limits = c(0, max(eig_df$pct_cumul) * 1.08),
    expand = expansion(mult = c(0, 0.02)),
    sec.axis = sec_axis(~ ., name = "% cumulé (courbe bleue)")
  ) +
  labs(
    title    = "Éboulis des valeurs propres — AFC lemmes × phases",
    subtitle = sprintf("%d dimensions | Dim1 = %.1f%% | Dim2 = %.1f%%",
                       nrow(eig_df), eig_df$pct_variance[1], eig_df$pct_variance[2]),
    x        = NULL,
    caption  = paste0(
      "Barres oranges = % variance par dimension. ",
      "Courbe bleue = % cumulé. ",
      "Tiret rouge = seuil Kaiser (100 / nb_dim)."
    )
  ) +
  theme_madyar()

save_plot(p_scree, file.path(OUT, "33d_afc_eboulis.png"),
          format = "wide_16_9", width = 9, dpi = 600)

# ---------------------------------------------------------------------------
# 5. GRAPHIQUE AFC — BIPLOT
# ---------------------------------------------------------------------------

coord_rows <- as.data.frame(res_ca$row$coord[top_lemmes, 1:2])
coord_rows$lemma <- rownames(coord_rows)
coord_rows$contrib <- contrib_total[top_lemmes]
mat_prop <- sweep(mat[top_lemmes, ], 1, rowSums(mat[top_lemmes, ]), "/")
coord_rows$phase_dom <- PHASES[apply(mat_prop, 1, which.max)]
coord_rows$phase_dom <- factor(coord_rows$phase_dom, levels = PHASES)

coord_cols <- as.data.frame(res_ca$col$coord[, 1:2])
coord_cols$phase <- rownames(coord_cols)

pct1 <- round(eig[1, "percentage of variance"], 1)
pct2 <- round(eig[2, "percentage of variance"], 1)

p_afc <- ggplot() +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.4) +
  geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.4) +
  geom_point(data = coord_rows,
             aes(x = `Dim 1`, y = `Dim 2`, colour = phase_dom, size = contrib),
             alpha = 0.75, shape = 16) +
  geom_text_repel(
    data = coord_rows,
    aes(x = `Dim 1`, y = `Dim 2`, label = lemma, colour = phase_dom),
    size = 2.8, max.overlaps = 40, segment.colour = "grey70",
    segment.size = 0.3, show.legend = FALSE
  ) +
  geom_point(data = coord_cols,
             aes(x = `Dim 1`, y = `Dim 2`),
             shape = 17, size = 5, colour = "grey10") +
  geom_text_repel(
    data = coord_cols,
    aes(x = `Dim 1`, y = `Dim 2`, label = phase),
    size = 3.5, fontface = "bold", colour = "grey10",
    nudge_y = 0.05, show.legend = FALSE
  ) +
  scale_colour_phase(short = TRUE) +
  scale_size_continuous(range = c(1.5, 5), guide = "none") +
  labs(
    title    = "AFC lemmes × phases — plan factoriel (Dim1 × Dim2)",
    subtitle = sprintf("Dim1 = %.1f%% — Dim2 = %.1f%% — Top %d lemmes par contribution",
                       pct1, pct2, N_TOP),
    x        = sprintf("Dim 1 (%.1f%%)", pct1),
    y        = sprintf("Dim 2 (%.1f%%)", pct2),
    colour   = "Phase dominante",
    caption  = paste0(
      "Source : lemmes_combined.csv — filtres : POS ∈ {NOUN, VERB, ADJ, PRON, ADV}, ",
      "n_total ≥ 30, présent dans ≥ 2 phases.\n",
      "Triangles noirs = barycentres des phases. Points = lemmes colorés par phase dominante."
    )
  ) +
  theme_madyar() +
  theme(legend.position = "bottom")

save_plot(p_afc, file.path(OUT, "33a_afc_lexicale_biplot.png"),
          format = "square", width = 11, dpi = 600)

# ---------------------------------------------------------------------------
# 6. GRAPHIQUE — CONTRIBUTIONS DIM1 & DIM2
# ---------------------------------------------------------------------------

contrib_df <- data.frame(
  lemma    = rownames(res_ca$row$contrib),
  contrib1 = res_ca$row$contrib[, 1],
  contrib2 = res_ca$row$contrib[, 2],
  coord1   = res_ca$row$coord[, 1],
  coord2   = res_ca$row$coord[, 2]
) %>%
  filter(lemma %in% top_lemmes) %>%
  mutate(
    side1 = if_else(coord1 > 0, "Dim1+", "Dim1−"),
    side2 = if_else(coord2 > 0, "Dim2+", "Dim2−")
  )

# Top 20 Dim1
top_dim1 <- contrib_df %>%
  arrange(desc(contrib1)) %>%
  slice_head(n = 20) %>%
  mutate(
    lemma = reorder(lemma, contrib1),
    fill  = if_else(coord1 > 0,
                    unname(PAL_PHASE["3_Institutionnel"]),
                    unname(PAL_PHASE["1_Artisanal"])),
    label_phase = if_else(coord1 > 0, "Institutionnel →", "← Artisanal")
  )

p_c1 <- ggplot(top_dim1, aes(x = lemma, y = contrib1, fill = fill)) +
  geom_col(width = 0.72) +
  geom_text(aes(y = contrib1 + max(contrib1) * 0.02,
                label = sprintf("%.2f%%", contrib1)),
            hjust = 0, size = 3, colour = "grey25") +
  coord_flip() +
  scale_fill_identity() +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.15)),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title    = "Top 20 lemmes — contribution à l'axe 1",
    subtitle = sprintf("Dim1 = %.1f%% de la variance totale | bleu = côté Artisanal (gauche), gris = côté Institutionnel (droite)", pct1),
    x        = NULL,
    y        = "Contribution (%)",
    caption  = "Couleur = phase dominante du côté du lemme sur Dim1."
  ) +
  theme_madyar() +
  theme(
    panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.3),
    panel.grid.major.y = element_blank()
  )

# Top 20 Dim2
top_dim2 <- contrib_df %>%
  arrange(desc(contrib2)) %>%
  slice_head(n = 20) %>%
  mutate(
    lemma = reorder(lemma, contrib2),
    fill  = if_else(coord2 > 0,
                    unname(PAL_PHASE["2_Semi-pro"]),
                    unname(PAL_PHASE["1_Artisanal"]))
  )

p_c2 <- ggplot(top_dim2, aes(x = lemma, y = contrib2, fill = fill)) +
  geom_col(width = 0.72) +
  geom_text(aes(y = contrib2 + max(contrib2) * 0.02,
                label = sprintf("%.2f%%", contrib2)),
            hjust = 0, size = 3, colour = "grey25") +
  coord_flip() +
  scale_fill_identity() +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.15)),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title    = "Top 20 lemmes — contribution à l'axe 2",
    subtitle = sprintf("Dim2 = %.1f%% de la variance totale | bleu = côté Artisanal (bas), orange = côté Semi-pro (haut)", pct2),
    x        = NULL,
    y        = "Contribution (%)",
    caption  = "Couleur = phase dominante du côté du lemme sur Dim2."
  ) +
  theme_madyar() +
  theme(
    panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.3),
    panel.grid.major.y = element_blank()
  )

save_plot(p_c1, file.path(OUT, "33b_afc_contrib_dim1.png"),
          format = "wide_16_9", width = 10, dpi = 600)
save_plot(p_c2, file.path(OUT, "33c_afc_contrib_dim2.png"),
          format = "wide_16_9", width = 10, dpi = 600)

# ---------------------------------------------------------------------------
# 7. EXPORT CSV COMPLET
# ---------------------------------------------------------------------------

# Coordonnées + contributions + cos2 de toutes les lignes (termes)
export_rows <- as.data.frame(res_ca$row$coord) %>%
  rename(coord_dim1 = `Dim 1`, coord_dim2 = `Dim 2`) %>%
  mutate(
    lemma    = rownames(.),
    contrib1 = res_ca$row$contrib[, 1],
    contrib2 = res_ca$row$contrib[, 2],
    cos2_1   = res_ca$row$cos2[, 1],
    cos2_2   = res_ca$row$cos2[, 2],
    in_top60 = lemma %in% top_lemmes
  ) %>%
  select(lemma, everything()) %>%
  arrange(desc(contrib1 + contrib2))

write_csv(export_rows, file.path(OUT, "33_afc_coordonnees_termes.csv"))

# Coordonnées + contributions + cos2 des colonnes (phases)
export_cols <- as.data.frame(res_ca$col$coord) %>%
  rename(coord_dim1 = `Dim 1`, coord_dim2 = `Dim 2`) %>%
  mutate(
    phase    = rownames(.),
    contrib1 = res_ca$col$contrib[, 1],
    contrib2 = res_ca$col$contrib[, 2],
    cos2_1   = res_ca$col$cos2[, 1],
    cos2_2   = res_ca$col$cos2[, 2]
  ) %>%
  select(phase, everything())

write_csv(export_cols, file.path(OUT, "33_afc_coordonnees_phases.csv"))

# Valeurs propres
export_eig <- as.data.frame(eig) %>%
  mutate(dimension = seq_len(nrow(eig))) %>%
  select(dimension, everything())
colnames(export_eig) <- c("dimension", "valeur_propre", "pct_variance", "pct_cumule")
write_csv(export_eig, file.path(OUT, "33_afc_valeurs_propres.csv"))

message("✔ Sauvé: 33_afc_coordonnees_termes.csv")
message("✔ Sauvé: 33_afc_coordonnees_phases.csv")
message("✔ Sauvé: 33_afc_valeurs_propres.csv")
message("\n=== Terminé : 4 PNG + 3 CSV dans ", OUT, " ===")
