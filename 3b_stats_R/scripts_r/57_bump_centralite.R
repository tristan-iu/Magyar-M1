# 57_bump_centralite.R — Bump chart évolution centralité (betweenness) P1→P2→P3
# Produit :
#   4_data_et_viz/57a_bump_centralite_caption.png
#   4_data_et_viz/57b_bump_centralite_dialogue.png
#   4_data_et_viz/57_bump_centralite_rangs.csv (synthèse caption + dialogue)
# Rscript 3b_stats_R/scripts_r/57_bump_centralite.R

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

suppressPackageStartupMessages(library(ggrepel))

COOC_DIR <- file.path(BASE, "..", "4_data_et_viz", "lexico")

# Paramètres
TOP_N    <- 15L                  # On garde les top-N lemmes par betweenness/phase
SENTINEL <- TOP_N + 1L           # Rang assigné aux lemmes hors top-N ("16+")
SEUIL_TENDANCE <- 5L             # |delta rang| pour qualifier emergent/declinant

# ---------------------------------------------------------------------------
# Lecture + calcul des rangs par phase
# ---------------------------------------------------------------------------

# On rang-ise par betweenness décroissant. Si betweenness = 0, le lemme n'est
# pas dans le réseau PMI de cette phase → rang sentinel SENTINEL.
# Tout lemme hors top-N est aussi écrasé sur SENTINEL pour visualiser sa "sortie".
calc_rang <- function(betw) {
  r <- rank(-betw, ties.method = "min")
  r[betw == 0] <- SENTINEL
  pmin(r, SENTINEL)
}

construire_df_bump <- function(source) {
  fichier <- file.path(COOC_DIR, sprintf("cooc_centralite_evolution_%s.csv", source))
  df <- read.csv(fichier, stringsAsFactors = FALSE, encoding = "UTF-8")

  df$rang_P1 <- calc_rang(df$P1_betweenness)
  df$rang_P2 <- calc_rang(df$P2_betweenness)
  df$rang_P3 <- calc_rang(df$P3_betweenness)

  # Filtre : top-N dans au moins une phase
  df <- df[df$rang_P1 <= TOP_N | df$rang_P2 <= TOP_N | df$rang_P3 <= TOP_N, ]

  # Tendance basée sur (rang_P1 - rang_P3). Rang inversé (1 = mieux) :
  # delta > 0 → progresse (passe d'un rang plus haut vers un rang plus bas).
  df$delta <- df$rang_P1 - df$rang_P3
  df$tendance <- ifelse(df$delta >=  SEUIL_TENDANCE, "emergent",
                 ifelse(df$delta <= -SEUIL_TENDANCE, "declinant", "stable"))
  df$source <- source
  df
}

df_cap <- construire_df_bump("caption")
df_dia <- construire_df_bump("dialogue")

message(sprintf("=== 57_bump_centralite.R ==="))
message(sprintf("  Caption : %d lemmes (top-%d dans \u22651 phase)", nrow(df_cap), TOP_N))
message(sprintf("    \u00b7 \u00e9mergents=%d, d\u00e9clinants=%d, stables=%d",
                sum(df_cap$tendance == "emergent"),
                sum(df_cap$tendance == "declinant"),
                sum(df_cap$tendance == "stable")))
message(sprintf("  Dialogue : %d lemmes", nrow(df_dia)))
message(sprintf("    \u00b7 \u00e9mergents=%d, d\u00e9clinants=%d, stables=%d",
                sum(df_dia$tendance == "emergent"),
                sum(df_dia$tendance == "declinant"),
                sum(df_dia$tendance == "stable")))

# ---------------------------------------------------------------------------
# Tracé du bump chart
# ---------------------------------------------------------------------------

# Palette tendance — vert/rouge/gris.
PAL_TENDANCE <- c(emergent  = "#2CA25F",
                  declinant = "#DE2D26",
                  stable    = "#888888")

tracer_bump <- function(df_wide, titre, sous_titre) {
  # On identifie les top trajectoires (notables, à mettre en gras)
  notables <- df_wide |>
    filter(tendance != "stable") |>
    arrange(desc(abs(delta))) |>
    slice_head(n = 6L) |>
    pull(lemma)

  # Reshape long pour ggplot
  df_long <- df_wide |>
    select(lemma, rang_P1, rang_P2, rang_P3, tendance, delta) |>
    tidyr::pivot_longer(cols = starts_with("rang_"),
                        names_to = "phase", values_to = "rang") |>
    mutate(phase = factor(sub("rang_", "", phase),
                          levels = c("P1", "P2", "P3")),
           notable = lemma %in% notables)

  # Labels : pour chaque lemme on étiquette son "meilleur" rang (= rang min).
  # Cela évite d'avoir 3 étiquettes par ligne et garde la lisibilité.
  df_labels <- df_long |>
    group_by(lemma) |>
    slice_min(rang, n = 1, with_ties = FALSE) |>
    ungroup()

  ggplot(df_long, aes(x = phase, y = rang, group = lemma, colour = tendance)) +
    geom_hline(yintercept = SENTINEL, colour = "grey80",
               linetype = "dotted", linewidth = 0.4) +
    annotate("text", x = 0.55, y = SENTINEL,
             label = paste0(TOP_N, "+ (hors r\u00e9seau)"),
             hjust = 0, vjust = -0.4, size = 2.8,
             colour = "grey50", fontface = "italic") +
    geom_line(aes(linewidth = notable, alpha = notable)) +
    geom_point(aes(size = notable)) +
    geom_text_repel(data = df_labels,
                    aes(label = lemma, fontface = ifelse(notable, "bold", "plain")),
                    size = 3.2, family = "sans",
                    box.padding = 0.25, point.padding = 0.15,
                    segment.size = 0.3, segment.colour = "grey70",
                    max.overlaps = Inf,
                    show.legend = FALSE) +
    scale_y_reverse(
      breaks = c(1, 5, 10, 15, SENTINEL),
      labels = c("1", "5", "10", "15", paste0(TOP_N, "+")),
      expand = expansion(mult = c(0.05, 0.08))
    ) +
    scale_x_discrete(expand = expansion(add = c(0.5, 0.5))) +
    scale_colour_manual(values = PAL_TENDANCE,
                        labels = c(emergent  = "\u00c9mergent (P1\u2192P3 monte)",
                                   declinant = "D\u00e9clinant (P1\u2192P3 chute)",
                                   stable    = "Stable / oscillant"),
                        name = NULL) +
    scale_linewidth_manual(values = c(`TRUE` = 1.3, `FALSE` = 0.5), guide = "none") +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.55), guide = "none") +
    scale_size_manual(values = c(`TRUE` = 3, `FALSE` = 1.8), guide = "none") +
    labs(title = titre, subtitle = sous_titre,
         x = NULL, y = "Rang par centralit\u00e9 (betweenness)",
         caption = paste0("Source : 4_data_et_viz/lexico/cooc_centralite_evolution_*.csv. ",
                          "Top-", TOP_N, " par phase. Sentinel ", TOP_N,
                          "+ = lemme hors r\u00e9seau PMI dans cette phase.")) +
    theme_madyar() +
    theme(legend.position = "bottom",
          panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3))
}

p_cap <- tracer_bump(df_cap,
  "Trajectoires des lemmes-pivots — caption",
  "Bump chart du rang par betweenness P1\u2192P2\u2192P3. Lignes \u00e9paisses = top 6 trajectoires (|\u0394 rang| max).")

p_dia <- tracer_bump(df_dia,
  "Trajectoires des lemmes-pivots — dialogue",
  "Bump chart du rang par betweenness P1\u2192P2\u2192P3. Lignes \u00e9paisses = top 6 trajectoires (|\u0394 rang| max).")

save_plot(p_cap, file.path(OUT, "57a_bump_centralite_caption.png"),
          format = "wide_16_9", width = 12)
save_plot(p_dia, file.path(OUT, "57b_bump_centralite_dialogue.png"),
          format = "wide_16_9", width = 12)

# ---------------------------------------------------------------------------
# CSV de synthèse — lemma × phase × (betweenness, rang) + tendance
# ---------------------------------------------------------------------------

df_synth <- bind_rows(
  df_cap |> mutate(source = "caption"),
  df_dia |> mutate(source = "dialogue")
) |>
  transmute(
    source, lemma,
    betw_P1 = P1_betweenness, rang_P1,
    betw_P2 = P2_betweenness, rang_P2,
    betw_P3 = P3_betweenness, rang_P3,
    delta_rang = delta, tendance
  ) |>
  arrange(source, desc(abs(delta_rang)))

write.csv(df_synth, file.path(OUT, "57_bump_centralite_rangs.csv"),
          row.names = FALSE)
message("\u2714 Sauv\u00e9 : 57_bump_centralite_rangs.csv (", nrow(df_synth), " lignes)")

message("=== 57_bump_centralite.R termin\u00e9 ===")
