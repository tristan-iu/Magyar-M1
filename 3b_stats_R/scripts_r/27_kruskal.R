# 27_kruskal.R — Kruskal-Wallis + post-hoc Dunn par phase
# Produit : 4_data_et_viz/27_kruskal_boxplots.png, 27_kruskal_tableau.csv, 27_dunn_posthoc.csv
# Rscript 3b_stats_R/scripts_r/27_kruskal.R
# Requiert : install.packages("rstatix")

library(stringi)

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
message("=== 27_kruskal.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------

compute_kruskal_dunn <- function(data,
                                 metrics = c("vues", "transferts", "reactions",
                                             "duree_sec", "caption_nchar"),
                                 p_adj_method = "BH") {
  if (!requireNamespace("rstatix", quietly = TRUE))
    stop("Package 'rstatix' requis : install.packages('rstatix')")
  library(rstatix)

  df_kw <- data %>%
    filter(!is.na(phase)) %>%
    mutate(
      phase         = factor(phase, levels = c(1, 2, 3), labels = c("P1", "P2", "P3")),
      caption_nchar = stri_length(coalesce(as.character(legende), ""))
    )

  .prep <- function(m) {
    d <- if (m == "duree_sec") df_kw %>% filter(media_type == "video", duree_sec > 0) else df_kw
    d %>% filter(!is.na(.data[[m]]))
  }
  .ok <- function(d) nrow(d) >= 6 && n_distinct(d$phase) >= 2

  kruskal_res <- lapply(metrics, function(m) {
    d <- .prep(m); if (!.ok(d)) return(NULL)
    tryCatch(rstatix::kruskal_test(d, as.formula(paste(m, "~ phase"))) %>% mutate(metric = m),
             error = function(e) NULL)
  })

  dunn_res <- lapply(metrics, function(m) {
    d <- .prep(m); if (!.ok(d)) return(NULL)
    tryCatch(rstatix::dunn_test(d, as.formula(paste(m, "~ phase")),
                                p.adjust.method = p_adj_method) %>% mutate(metric = m),
             error = function(e) NULL)
  })

  list(
    kruskal = bind_rows(Filter(Negate(is.null), kruskal_res)),
    dunn    = bind_rows(Filter(Negate(is.null), dunn_res))
  )
}

plot_kruskal_boxplots <- function(data,
                                  metrics = c("vues", "transferts", "reactions",
                                              "duree_sec", "caption_nchar"),
                                  ncol = 3,
                                  titre = "Distribution par phase (Kruskal-Wallis)") {
  if (!requireNamespace("rstatix", quietly = TRUE))
    stop("Package 'rstatix' requis.")
  library(rstatix)

  df_base <- data %>%
    filter(!is.na(phase)) %>%
    mutate(
      phase         = factor(phase, levels = c(1, 2, 3), labels = c("P1", "P2", "P3")),
      caption_nchar = stri_length(coalesce(as.character(legende), ""))
    )

  .prep <- function(m) {
    d <- if (m == "duree_sec") df_base %>% filter(media_type == "video", duree_sec > 0) else df_base
    d %>% filter(!is.na(.data[[m]]))
  }

  kw_pvals <- lapply(metrics, function(m) {
    d <- .prep(m)
    if (nrow(d) < 6 || n_distinct(d$phase) < 2) return(tibble(metric = m, pval_label = "n.d."))
    tryCatch({
      res <- rstatix::kruskal_test(d, as.formula(paste(m, "~ phase")))
      tibble(metric = m, pval_label = paste0("KW p=", signif(res$p, 2)))
    }, error = function(e) tibble(metric = m, pval_label = "erreur"))
  }) %>% bind_rows()

  labels_metric <- c(
    vues = "Vues", transferts = "Forwards", reactions = "Réactions",
    duree_sec = "Durée vidéo (s)", caption_nchar = "Longueur legende"
  )

  df_long <- bind_rows(lapply(metrics, function(m) {
    .prep(m) %>% transmute(phase, value = .data[[m]], metric = m)
  })) %>%
    left_join(kw_pvals, by = "metric") %>%
    mutate(
      facet_label = paste0(labels_metric[metric], "\n", pval_label),
      facet_label = factor(facet_label,
        levels = paste0(labels_metric[metrics], "\n",
                        kw_pvals$pval_label[match(metrics, kw_pvals$metric)]))
    )

  pal_phase_box <- c(
    P1 = unname(PAL_PHASE["1_Artisanal"]),
    P2 = unname(PAL_PHASE["2_Semi-pro"]),
    P3 = unname(PAL_PHASE["3_Institutionnel"])
  )

  ggplot(df_long, aes(x = phase, y = value, fill = phase)) +
    geom_boxplot(outlier.shape = NA, width = 0.5) +
    geom_jitter(width = 0.15, alpha = 0.15, size = 0.8) +
    facet_wrap(~facet_label, ncol = ncol, scales = "free_y") +
    scale_fill_manual(values = pal_phase_box) +
    labs(
      title    = titre,
      subtitle = "Boxplots par phase (outliers masqués) — jitter alpha=0.15",
      x = "Phase", y = NULL,
      caption  = "Tests de Kruskal-Wallis (non-paramétriques). Correction BH pour post-hoc Dunn."
    ) +
    theme_madyar_facet() +
    theme(legend.position = "none")
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

metrics_kw <- c("vues", "transferts", "reactions", "duree_sec", "caption_nchar")

kw_results <- compute_kruskal_dunn(df_clean, metrics = metrics_kw, p_adj_method = "BH")

write.csv(kw_results$kruskal, file.path(OUT, "27_kruskal_tableau.csv"), row.names = FALSE)
message("\u2714 Sauvé: 27_kruskal_tableau.csv")

write.csv(kw_results$dunn, file.path(OUT, "27_dunn_posthoc.csv"), row.names = FALSE)
message("\u2714 Sauvé: 27_dunn_posthoc.csv")

save_plot(
  plot_kruskal_boxplots(df_clean, metrics = metrics_kw, ncol = 3),
  file.path(OUT, "27_kruskal_boxplots.png"),
  format = "wide_16_9", width = 14, dpi = 600
)

message("=== Terminé : 1 PNG + 2 CSV dans ", OUT, " ===")
