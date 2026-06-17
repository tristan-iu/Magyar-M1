# 58_volcano_phases.R — Volcano plots inter-phases (P1↔P2, P2↔P3, P1↔P3)
# Produit :
#   4_data_et_viz/58a_volcano_caption.png      (3 facets, lemmes uk)
#   4_data_et_viz/58b_volcano_caption_fr.png   (3 facets, lemmes traduits)
#   4_data_et_viz/58c_volcano_dialogue.png
#   4_data_et_viz/58d_volcano_dialogue_fr.png
#   4_data_et_viz/58_volcano_top_lemmes.csv
# Rscript 3b_stats_R/scripts_r/58_volcano_phases.R

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

LEXICO_DIR <- file.path(BASE, "..", "4_data_et_viz", "lexico")

# Paramètres
SEUIL_LOG2FC <- 1.0       # |log2FC| pour considérer une différence "biologiquement" notable
SEUIL_PADJ   <- 0.05      # p-value ajustée (Benjamini-Hochberg)
COMPTE_MIN   <- 5L        # count_a + count_b minimum pour éviter les hapax bruyants
TOP_N_LABELS <- 10L       # Lemmes annotés par facet (les plus "volcano-significatifs")

# ---------------------------------------------------------------------------
# Dictionnaire de traduction uk→fr (port R). Si le lemme n'est pas mappé,
# on renvoie le lemme uk inchangé.
# ---------------------------------------------------------------------------

TRADUCTIONS <- c(
  "\u0431\u0430\u0445\u043c\u0443\u0442"      = "Bakhmout",
  "\u0431\u0430\u0447\u0438\u0442\u0438"      = "voir",
  "\u0431\u0440\u0438\u0433\u0430\u0434\u0430" = "brigade",
  "\u0431\u0440\u043e\u0432\u0434\u0456"      = "Brovdi",
  "\u0432\u0435\u043b\u0438\u043a\u0438\u0439" = "grand",
  "\u0432\u0438\u044f\u0432\u0438\u0442\u0438" = "d\u00e9tecter",
  "\u0432\u0439\u043e"                          = "allez !",
  "\u0432\u043e\u0440\u043e\u0433"             = "ennemi",
  "\u0432\u0456\u0434\u0435\u043e"             = "vid\u00e9o",
  "\u0432\u0456\u0439\u0441\u044c\u043a\u043e\u0432\u0438\u0439" = "militaire",
  "\u0433\u0440\u043d"                          = "hryvnia",
  "\u0434\u0438\u0432\u0438\u0442\u0438\u0441\u044f" = "regarder",
  "\u0434\u043e\u0431\u0430"                    = "24 h",
  "\u0434\u0440\u043e\u043d"                    = "drone",
  "\u0434\u044f\u043a\u0443\u0432\u0430\u0442\u0438" = "remercier",
  "\u0436\u0430\u043b\u043e"                    = "dard / FPV",
  "\u0436\u0430\u0445"                          = "horreur",
  "\u0437\u0430\u0441\u0456\u0431"              = "moyen / engin",
  "\u0437\u0431\u0456\u0440"                    = "collecte",
  "\u0437\u043d\u0430\u0442\u0438"              = "savoir",
  "\u0437\u043d\u0438\u0449\u0438\u0442\u0438"  = "d\u00e9truire",
  "\u0437\u0441\u0443"                          = "ZSU",
  "\u0439\u043e\u0431\u043b\u0438\u043a\u0456\u0432" = "argot FPV",
  "\u043a\u0430\u0437\u0430\u0442\u0438"        = "dire",
  "\u043a\u0430\u0447\u0430\u0442\u0438"        = "t\u00e9l\u00e9charger",
  "\u043a\u043e\u0436\u043d\u0438\u0439"        = "chaque",
  "\u043a\u043e\u0442\u0440\u0438\u0439"        = "lequel/qui",
  "\u043a\u0440\u0438\u043d\u043a\u0438"        = "Krynky",
  "\u043a\u0456\u043b\u044c\u043a\u0456\u0441\u0442\u044c" = "quantit\u00e9",
  "\u043b\u044e\u0434\u0438\u043d\u0430"        = "personne",
  "\u043c\u0430\u0434\u044f\u0440"              = "Magyar",
  "\u043c\u0430\u043a\u0456\u0442\u0440\u0430"  = "t\u00eate (argot)",
  "\u043c\u043b\u043d"                          = "million",
  "\u043c\u043e\u043d\u043e"                    = "Monobank",
  "\u043c\u043e\u0440\u0441\u044c\u043a\u0438\u0439" = "maritime",
  "\u043c\u0456\u0441\u044f\u0446\u044c"        = "mois",
  "\u043d\u0430\u0441\u0442\u0443\u043f\u043d\u0438\u0439" = "suivant",
  "\u043d\u0456\u0447\u043d\u0438\u0439"        = "nocturne",
  "\u043e\u0431\u0443\u0431\u0430\u0441"        = "UBAS",
  "\u043e\u043a\u043e"                          = "\u0153il",
  "\u043e\u043a\u0440\u0435\u043c\u0438\u0439"  = "s\u00e9par\u00e9 / distinct",
  "\u043e\u0442"                                = "donc",
  "\u043e\u0442\u0440\u0438\u043c\u0443\u0432\u0430\u0447" = "destinataire",
  "\u043f\u043c"                                = "Ptakhyky Madyara",
  "\u043f\u0440\u0430\u0446\u044e\u0432\u0430\u0442\u0438" = "travailler",
  "\u043f\u0440\u0438\u0432\u0430\u0442\u0431\u0430\u043d\u043a" = "PrivatBank",
  "\u043f\u0440\u043e\u0442\u044f\u0433\u043e\u043c"          = "pendant",
  "\u043f\u0442\u0430\u0445"                    = "oiseau / drone",
  "\u043f\u0456\u0434\u0440\u043e\u0437\u0434\u0456\u043b" = "unit\u00e9",
  "\u043f\u0456\u043b\u043e\u0442"              = "pilote",
  "\u0440\u0435\u043a\u0432\u0456\u0437\u0438\u0442" = "coord. bancaires",
  "\u0440\u0435\u0440"                          = "guerre \u00e9lec.",
  "\u0440\u043e\u0431\u0435\u0440\u0442"        = "Robert",
  "\u0440\u043e\u0431\u0438\u0442\u0438"        = "faire",
  "\u0440\u0456\u043a"                          = "ann\u00e9e",
  "\u0441\u0431\u0441"                          = "SBS",
  "\u0441\u0432\u0456\u0439"                    = "son propre",
  "\u0441\u043f\u043e\u043a\u0456\u0439\u043d\u0438\u0439" = "calme",
  "\u0442\u0438\u0441\u044f\u0447\u0430"        = "mille",
  "\u0442\u0438\u0445\u0438\u0439"              = "silencieux",
  "\u0442\u043e\u0436"                          = "donc/alors",
  "\u0442\u0456\u043a\u0442\u043e\u0446\u0456"  = "TikTok",
  "\u0443\u043a\u0440\u0430\u0457\u043d\u0430"  = "Ukraine",
  "\u0444\u043f\u0432"                          = "FPV",
  "\u0445\u0440\u043e\u0431\u0430\u043a"        = "ver / FPV",
  "\u0445\u0440\u043e\u0431\u0430\u0447\u0438\u0439" = "ver (adj.)",
  "\u0445\u0440\u043e\u0431\u0430\u0447\u0443"  = "ver (acc.)",
  "\u0446\u0456\u043b\u044c"                    = "cible",
  "\u044f\u043a\u0438\u0439\u0441\u044c"        = "un certain"
)

traduire_lemme <- function(x) {
  ifelse(x %in% names(TRADUCTIONS), TRADUCTIONS[x], x)
}

# ---------------------------------------------------------------------------
# Pré-traitement commun
# ---------------------------------------------------------------------------

# La colonne `comparison` du CSV vaut p.ex. "P1_Artisanal → P2_Semi-pro".
# On extrait la paire courte ("P1", "P2") et on l'utilise comme facet label.
parser_comparaison <- function(comp) {
  # Découpage sur la flèche ; les deux moitiés sont "Pn_Label".
  parts <- strsplit(comp, " \u2192 ", fixed = TRUE)
  do.call(rbind, lapply(parts, function(p) {
    g <- sub("_.*$", "", p)
    data.frame(gauche = g[1], droite = g[2], stringsAsFactors = FALSE)
  }))
}

preparer <- function(source) {
  fichier <- file.path(LEXICO_DIR, sprintf("volcano_%s.csv", source))
  df <- read.csv(fichier, stringsAsFactors = FALSE, encoding = "UTF-8")

  # Sécurité : pvalue_adj est cappée à 1 → neg_log10_padj peut être -0.
  df$neg_log10_padj <- pmax(df$neg_log10_padj, 0)

  # Paires de phases (gauche/droite) pour chaque comparaison.
  paires <- parser_comparaison(df$comparison)
  df$phase_gauche <- paires$gauche
  df$phase_droite <- paires$droite

  # Phase d'enrichissement par ligne :
  #   log2fc > 0 → enrichi côté droite ; log2fc < 0 → enrichi côté gauche.
  #   Non-significatif (NS) si |log2fc| < seuil OU p_adj ≥ seuil OU compte insuffisant.
  df$significatif <- df$pvalue_adj < SEUIL_PADJ &
                     abs(df$log2fc) >= SEUIL_LOG2FC &
                     (df$count_a + df$count_b) >= COMPTE_MIN
  df$phase_enrichie <- ifelse(!df$significatif, "NS",
                              ifelse(df$log2fc > 0,
                                     paste0("P", substr(df$phase_droite, 2, 2)),
                                     paste0("P", substr(df$phase_gauche, 2, 2))))

  # Label de facet : "P1 → P2"
  df$facet_label <- paste0(df$phase_gauche, " \u2192 ", df$phase_droite)
  df$facet_label <- factor(df$facet_label,
                           levels = c("P1 \u2192 P2", "P2 \u2192 P3", "P1 \u2192 P3"))

  # Score volcano (pour le top des labels) : combine significance et amplitude.
  df$score_volcano <- df$neg_log10_padj * abs(df$log2fc)

  df$compte_total <- df$count_a + df$count_b
  df$source <- source
  df
}

# ---------------------------------------------------------------------------
# Tracé du volcano (3 facets)
# ---------------------------------------------------------------------------

# Palette = phases du projet + gris pour NS. On réutilise PAL_PHASE qui est
# défini en clé longue ("1_Artisanal", etc.), on aliase en P1/P2/P3 pour
# coller à phase_enrichie.
PAL_VOLCANO <- c(
  P1 = unname(PAL_PHASE["1_Artisanal"]),
  P2 = unname(PAL_PHASE["2_Semi-pro"]),
  P3 = unname(PAL_PHASE["3_Institutionnel"]),
  NS = "grey75"
)

tracer_volcano <- function(df, source, traduire = FALSE) {

  # On sélectionne les top-N lemmes par facet pour annotation.
  # Seulement parmi les significatifs (sinon on annote du bruit).
  df_labels <- df |>
    filter(significatif) |>
    group_by(facet_label) |>
    slice_max(score_volcano, n = TOP_N_LABELS, with_ties = FALSE) |>
    ungroup() |>
    mutate(label = if (traduire) traduire_lemme(lemma) else lemma,
           # On marque visuellement les lemmes traduits depuis le dico
           # (italique si traduit) vs ceux laissés en uk (romain).
           est_traduit = lemma %in% names(TRADUCTIONS))

  # On trace les points NS en premier (couche basse, gris) puis les
  # significatifs par-dessus (sinon le rouge/bleu se fait avaler par le gris).
  ggplot(df, aes(x = log2fc, y = neg_log10_padj)) +
    # Seuils visuels
    geom_hline(yintercept = -log10(SEUIL_PADJ),
               colour = "firebrick", linetype = "dotted", linewidth = 0.4) +
    geom_vline(xintercept = c(-SEUIL_LOG2FC, SEUIL_LOG2FC),
               colour = "grey60", linetype = "dotted", linewidth = 0.4) +
    # Points non-significatifs (couche basse)
    geom_point(data = filter(df, !significatif),
               aes(size = compte_total),
               colour = PAL_VOLCANO["NS"], alpha = 0.35) +
    # Points significatifs (couche haute, colorés par phase)
    geom_point(data = filter(df, significatif),
               aes(size = compte_total, colour = phase_enrichie),
               alpha = 0.8) +
    # Annotations top lemmes
    geom_text_repel(data = df_labels,
                    aes(label = label,
                        fontface = ifelse(traduire & !est_traduit, "italic", "plain")),
                    size = 3, family = "sans",
                    box.padding = 0.3, point.padding = 0.2,
                    segment.size = 0.3, segment.colour = "grey60",
                    max.overlaps = Inf,
                    min.segment.length = 0,
                    show.legend = FALSE) +
    facet_wrap(~ facet_label, nrow = 1) +
    scale_colour_manual(values = PAL_VOLCANO, name = "Enrichi en",
                        breaks = c("P1", "P2", "P3"),
                        labels = c("P1 (Artisanal)",
                                   "P2 (Semi-pro)",
                                   "P3 (Institutionnel)")) +
    scale_size_continuous(range = c(0.6, 4.5), name = "Fr\u00e9quence brute",
                          breaks = c(10, 50, 200, 800),
                          trans = "sqrt") +
    scale_x_continuous(name = "log2 fold-change",
                       breaks = scales::pretty_breaks(n = 6)) +
    scale_y_continuous(name = "-log10(p-value ajust\u00e9e BH)",
                       breaks = scales::pretty_breaks(n = 6)) +
    labs(
      title = paste0(
        "Lemmes diff\u00e9renciateurs entre phases \u2014 ",
        source,
        if (traduire) " (lemmes traduits)" else ""
      ),
      subtitle = paste0(
        "Volcano plots : log2FC vs -log10(p_adj). ",
        "Seuils : |log2FC| \u2265 ", SEUIL_LOG2FC,
        ", p_adj < ", SEUIL_PADJ,
        ", compte\u22a5", COMPTE_MIN, ".",
        if (traduire) " Italique = lemme non pr\u00e9sent dans le dictionnaire (rest\u00e9 en uk)." else ""
      ),
      caption = paste0(
        "Source : 4_data_et_viz/lexico/volcano_", source, ".csv. ",
        "Test exact de Fisher + correction Benjamini-Hochberg. ",
        "Top ", TOP_N_LABELS, " lemmes annot\u00e9s par facet (score = -log10(p_adj)\u00d7|log2FC|)."
      )
    ) +
    theme_madyar() +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = element_text(face = "bold", size = 11),
          panel.spacing.x = unit(1, "lines"))
}

# ---------------------------------------------------------------------------
# Boucle : 2 sources × 2 modes (uk/fr) = 4 PNG
# ---------------------------------------------------------------------------

df_cap <- preparer("caption")
df_dia <- preparer("dialogue")

message("=== 58_volcano_phases.R ===")
for (src in list(list(d = df_cap, n = "caption"), list(d = df_dia, n = "dialogue"))) {
  message(sprintf("  %s : %d lemmes, %d significatifs (|log2FC|\u2265%g, p_adj<%g, compte\u2265%d)",
                  src$n, nrow(src$d), sum(src$d$significatif),
                  SEUIL_LOG2FC, SEUIL_PADJ, COMPTE_MIN))
  par_facet <- src$d |>
    filter(significatif) |>
    count(facet_label, phase_enrichie)
  print(par_facet)
}

p_cap_uk <- tracer_volcano(df_cap, "caption",  traduire = FALSE)
p_cap_fr <- tracer_volcano(df_cap, "caption",  traduire = TRUE)
p_dia_uk <- tracer_volcano(df_dia, "dialogue", traduire = FALSE)
p_dia_fr <- tracer_volcano(df_dia, "dialogue", traduire = TRUE)

save_plot(p_cap_uk, file.path(OUT, "58a_volcano_caption.png"),
          format = "wide_16_9", width = 14)
save_plot(p_cap_fr, file.path(OUT, "58b_volcano_caption_fr.png"),
          format = "wide_16_9", width = 14)
save_plot(p_dia_uk, file.path(OUT, "58c_volcano_dialogue.png"),
          format = "wide_16_9", width = 14)
save_plot(p_dia_fr, file.path(OUT, "58d_volcano_dialogue_fr.png"),
          format = "wide_16_9", width = 14)

# ---------------------------------------------------------------------------
# CSV — top lemmes par comparaison (citable directement dans le mémoire)
# ---------------------------------------------------------------------------

df_top <- bind_rows(
  df_cap |> mutate(source = "caption"),
  df_dia |> mutate(source = "dialogue")
) |>
  filter(significatif) |>
  group_by(source, facet_label) |>
  slice_max(score_volcano, n = 20L, with_ties = FALSE) |>
  ungroup() |>
  mutate(lemma_fr = traduire_lemme(lemma)) |>
  transmute(
    source,
    comparaison = facet_label,
    lemma, lemma_fr,
    log2fc = round(log2fc, 3),
    pvalue_adj = pvalue_adj,
    neg_log10_padj = round(neg_log10_padj, 3),
    count_a, count_b,
    phase_enrichie
  ) |>
  arrange(source, comparaison, desc(abs(log2fc)))

write.csv(df_top, file.path(OUT, "58_volcano_top_lemmes.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("\u2714 Sauv\u00e9 : 58_volcano_top_lemmes.csv (", nrow(df_top), " lignes)")

message("=== 58_volcano_phases.R termin\u00e9 ===")
