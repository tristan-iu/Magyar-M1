# 56_reseaux_pmi.R — Réseaux lexicaux PMI par phase (caption + dialogue)
# Produit :
#   4_data_et_viz/56a_reseau_pmi_caption.png       (3 panneaux côte-à-côte)
#   4_data_et_viz/56b_reseau_pmi_dialogue.png      (3 panneaux côte-à-côte)
#   4_data_et_viz/56c_centralite_top20_caption.png (bar chart par phase)
#   4_data_et_viz/56d_centralite_top20_dialogue.png
#   4_data_et_viz/56_reseau_pmi_<src>_<P>.gexf     (6 fichiers Gephi)
#   4_data_et_viz/56_reseau_pmi_<src>_<P>.html     (6 fichiers visNetwork interactif)
# Rscript 3b_stats_R/scripts_r/56_reseaux_pmi.R

this_file <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  if (length(f) > 0 && nzchar(f)) return(normalizePath(f, mustWork = FALSE))
  for (env in rev(sys.frames()))
    if (!is.null(env$ofile)) return(normalizePath(env$ofile, mustWork = FALSE))
  stop("Lancer via Rscript ou source().")
})
# Le script vit dans 3b_stats_R/scripts_r/ — 3 dirnames pour atteindre la racine du dépôt
REPO <- dirname(dirname(dirname(this_file)))
source(file.path(REPO, "3b_stats_R", "r_source.R"))
OUT  <- file.path(REPO, "4_data_et_viz")
dir.create(OUT, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(igraph)
  library(visNetwork)
})

# CSV produits par 3a_lexicometrie/cooccurrences.py --per-phase (défaut : 4_data_et_viz/lexico)
COOC_DIR <- file.path(REPO, "4_data_et_viz", "lexico")
set.seed(42)

# ---------------------------------------------------------------------------
# Lecture des CSV cooc + construction d'un igraph par (source, phase)
# ---------------------------------------------------------------------------

charger_graphe <- function(source, phase) {
  nodes <- read.csv(file.path(COOC_DIR, sprintf("cooc_nodes_%s_%s.csv", source, phase)),
                    stringsAsFactors = FALSE, encoding = "UTF-8")
  edges <- read.csv(file.path(COOC_DIR, sprintf("cooc_edges_%s_%s.csv", source, phase)),
                    stringsAsFactors = FALSE, encoding = "UTF-8")

  # On garde uniquement les arêtes dont les deux extrémités sont déclarées en nœuds
  nodes <- nodes[nchar(nodes$Id) > 0, ]
  edges <- edges[edges$Source %in% nodes$Id & edges$Target %in% nodes$Id, ]

  g <- graph_from_data_frame(d = edges[, c("Source", "Target", "Weight", "RawCount")],
                             vertices = nodes, directed = FALSE)

  # Communautés Louvain (poids = PMI). Couleur cohérente entre PNG / HTML / GEXF.
  V(g)$community <- as.integer(membership(cluster_louvain(g, weights = E(g)$Weight)))
  g
}

# ---------------------------------------------------------------------------
# Export GEXF manuel (Gephi-ready) — pas besoin de rgexf (XML simple)
# ---------------------------------------------------------------------------

ecrire_gexf <- function(g, fichier) {
  vs <- igraph::as_data_frame(g, what = "vertices")
  es <- igraph::as_data_frame(g, what = "edges")

  # Échappement XML basique pour les labels (cyrillique passe en UTF-8)
  xml_escape <- function(s) {
    s <- gsub("&", "&amp;",  s, fixed = TRUE)
    s <- gsub("<", "&lt;",   s, fixed = TRUE)
    s <- gsub(">", "&gt;",   s, fixed = TRUE)
    s <- gsub("\"", "&quot;", s, fixed = TRUE)
    s
  }

  nodes_xml <- paste0(
    '      <node id="', xml_escape(vs$name),
    '" label="', xml_escape(vs$Label), '">',
    '<attvalues>',
    '<attvalue for="freq" value="',         vs$Freq,        '"/>',
    '<attvalue for="degree" value="',       vs$Degree,      '"/>',
    '<attvalue for="betweenness" value="',  vs$Betweenness, '"/>',
    '<attvalue for="community" value="',    vs$community,   '"/>',
    '</attvalues></node>',
    collapse = "\n"
  )
  edges_xml <- paste0(
    '      <edge id="', seq_len(nrow(es)) - 1L,
    '" source="', xml_escape(es$from),
    '" target="', xml_escape(es$to),
    '" weight="', es$Weight, '"/>',
    collapse = "\n"
  )

  body <- paste0(
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n',
    '  <graph mode="static" defaultedgetype="undirected">\n',
    '    <attributes class="node">\n',
    '      <attribute id="freq" title="freq" type="integer"/>\n',
    '      <attribute id="degree" title="degree" type="integer"/>\n',
    '      <attribute id="betweenness" title="betweenness" type="double"/>\n',
    '      <attribute id="community" title="community" type="integer"/>\n',
    '    </attributes>\n',
    '    <nodes>\n', nodes_xml, '\n    </nodes>\n',
    '    <edges>\n', edges_xml, '\n    </edges>\n',
    '  </graph>\n',
    '</gexf>\n'
  )
  writeLines(body, fichier, useBytes = TRUE)
}

# ---------------------------------------------------------------------------
# Export HTML interactif (visNetwork) — drag/zoom/hover
# ---------------------------------------------------------------------------

ecrire_html <- function(g, fichier, titre) {
  vs <- igraph::as_data_frame(g, what = "vertices")
  es <- igraph::as_data_frame(g, what = "edges")

  vn_nodes <- data.frame(
    id    = vs$name,
    label = vs$Label,
    value = vs$Degree,
    title = sprintf("<b>%s</b><br>freq=%d<br>degree=%d<br>betweenness=%.2f",
                    vs$Label, vs$Freq, vs$Degree, vs$Betweenness),
    group = vs$community,
    stringsAsFactors = FALSE
  )
  vn_edges <- data.frame(
    from  = es$from,
    to    = es$to,
    value = es$Weight,
    title = sprintf("PMI=%.2f (n=%d)", es$Weight, es$RawCount),
    stringsAsFactors = FALSE
  )

  vn <- visNetwork(vn_nodes, vn_edges, main = titre, width = "100%", height = "750px") |>
    visIgraphLayout(layout = "layout_with_fr", randomSeed = 42) |>
    visNodes(scaling = list(min = 5, max = 30)) |>
    visEdges(color = list(opacity = 0.4)) |>
    visOptions(highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
               nodesIdSelection = TRUE) |>
    visLayout(randomSeed = 42)

  visSave(vn, fichier, selfcontained = TRUE)
}

# ---------------------------------------------------------------------------
# Tracé d'un panneau PNG via igraph base (1 sous-graphe par phase)
# ---------------------------------------------------------------------------

# Palette communautés (recyclage de PAL_CAT défini dans r_source.R)
couleur_communaute <- function(comm_ids) {
  pal <- PAL_CAT
  if (length(unique(comm_ids)) > length(pal)) {
    pal <- colorRampPalette(PAL_CAT)(length(unique(comm_ids)))
  }
  pal[match(comm_ids, sort(unique(comm_ids)))]
}

tracer_igraph <- function(g, titre, top_n_labels = 25) {
  bet <- V(g)$Betweenness
  rang_bet <- rank(-bet, ties.method = "min")
  labels <- ifelse(rang_bet <= top_n_labels, V(g)$Label, "")

  set.seed(42)
  lay <- layout_with_fr(g, weights = E(g)$Weight)

  plot(g,
       layout = lay,
       vertex.size = pmax(2, sqrt(V(g)$Degree) * 2),
       vertex.color = couleur_communaute(V(g)$community),
       vertex.frame.color = "white",
       vertex.label = labels,
       vertex.label.cex = 0.75,
       vertex.label.family = "sans",
       vertex.label.color = "grey15",
       vertex.label.dist = 0.4,
       edge.width = pmin(2, E(g)$Weight / 2),
       edge.color = adjustcolor("grey60", alpha.f = 0.35),
       margin = c(0, 0, 0, 0))
  title(main = titre, cex.main = 1.3, font.main = 2)
}

# ---------------------------------------------------------------------------
# Boucle de construction + export pour chaque (source, phase)
# ---------------------------------------------------------------------------

graphes <- list()
for (src in c("caption", "dialogue")) {
  graphes[[src]] <- list()
  for (ph in c("P1", "P2", "P3")) {
    g <- charger_graphe(src, ph)
    graphes[[src]][[ph]] <- g
    message(sprintf("  %s %s : %d nœuds, %d arêtes, %d communautés",
                    src, ph, vcount(g), ecount(g),
                    length(unique(V(g)$community))))

    ecrire_gexf(g, file.path(OUT, sprintf("56_reseau_pmi_%s_%s.gexf", src, ph)))
    ecrire_html(g, file.path(OUT, sprintf("56_reseau_pmi_%s_%s.html", src, ph)),
                sprintf("Réseau PMI — %s %s", src, ph))
  }
}
message("✔ GEXF + HTML exportés pour 6 (source × phase)")

# ---------------------------------------------------------------------------
# 56a / 56b — 3 panneaux PNG (caption puis dialogue)
# ---------------------------------------------------------------------------

for (src in c("caption", "dialogue")) {
  fichier <- file.path(OUT, sprintf("56%s_reseau_pmi_%s.png",
                                     if (src == "caption") "a" else "b", src))
  # Cairo : nécessaire pour rendre les caractères cyrilliques en PNG
  png(fichier, width = 18, height = 7, units = "in", res = 300,
      type = "cairo", bg = "white")
  layout(matrix(c(1, 2, 3, 4, 4, 4), nrow = 2, byrow = TRUE),
         heights = c(8, 1))
  par(mar = c(0.5, 0.5, 2, 0.5))

  for (ph in c("P1", "P2", "P3")) {
    lbl <- switch(ph, P1 = "P1 — Artisanal", P2 = "P2 — Semi-pro",
                  P3 = "P3 — Institutionnel")
    tracer_igraph(graphes[[src]][[ph]], lbl)
  }

  # Bandeau titre/légende en bas
  par(mar = c(0, 0, 0, 0))
  plot.new()
  seuil_pmi <- if (src == "caption") "2.0" else "1.0"
  text(0.5, 0.7,
       sprintf("Réseaux lexicaux PMI — %s", src),
       cex = 1.4, font = 2)
  text(0.5, 0.3,
       sprintf("Nœud = lemme, taille = degré, couleur = communauté Louvain. Arêtes PMI \u2265 %s. Top 25 lemmes par betweenness étiquetés.",
               seuil_pmi),
       cex = 0.9, col = "grey30")

  dev.off()
  message("✔ Sauvé : ", basename(fichier))
}

# ---------------------------------------------------------------------------
# 56c / 56d — Top-20 betweenness facetté par phase (ggplot)
# ---------------------------------------------------------------------------

for (src in c("caption", "dialogue")) {
  df_top <- do.call(rbind, lapply(c("P1", "P2", "P3"), function(ph) {
    nodes <- read.csv(file.path(COOC_DIR, sprintf("cooc_nodes_%s_%s.csv", src, ph)),
                      stringsAsFactors = FALSE, encoding = "UTF-8")
    nodes <- nodes[order(-nodes$Betweenness), ][1:min(20, nrow(nodes)), ]
    nodes$phase_lbl <- factor(switch(ph,
                                     P1 = "1_Artisanal",
                                     P2 = "2_Semi-pro",
                                     P3 = "3_Institutionnel"),
                              levels = names(PAL_PHASE))
    nodes$rang <- seq_len(nrow(nodes))
    nodes
  }))

  # On crée un Id unique pour préserver l'ordre dans le facet (sinon ggplot
  # réordonne globalement et casse l'effet "top par phase")
  df_top$label_id <- paste(df_top$phase_lbl, df_top$Label, sep = "__")
  df_top$label_id <- factor(df_top$label_id,
                            levels = rev(df_top$label_id[order(df_top$phase_lbl,
                                                                df_top$Betweenness)]))

  p <- ggplot(df_top, aes(x = Betweenness, y = label_id, fill = phase_lbl)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = Label), hjust = -0.1, size = 3, family = "sans") +
    facet_wrap(~ phase_lbl, scales = "free_y", nrow = 1,
               labeller = labeller(phase_lbl = LBL_PHASE_SHORT)) +
    scale_y_discrete(labels = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
    scale_fill_phase(short = TRUE) +
    labs(
      title = sprintf("Top 20 lemmes par centralité (betweenness) — %s", src),
      subtitle = "Lemmes qui ponctent le plus de plus courts chemins dans le réseau PMI.",
      x = "Betweenness centrality", y = NULL,
      caption = "Source : 4_data_et_viz/lexico/cooc_nodes_*."
    ) +
    theme_madyar_facet() +
    theme(axis.ticks.y = element_blank())

  save_plot(p, file.path(OUT, sprintf("56%s_centralite_top20_%s.png",
                                       if (src == "caption") "c" else "d", src)),
            format = "wide_16_9", width = 14)
}

message("=== 56_reseaux_pmi.R terminé ===")
