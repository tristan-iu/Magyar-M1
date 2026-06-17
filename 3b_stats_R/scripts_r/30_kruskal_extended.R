# 30_kruskal_extended.R — Kruskal-Wallis + Dunn étendu
# Produit : 4_data_et_viz/30_kruskal_extended_boxplots.png, 30_kruskal_extended.csv, 30_dunn_extended.csv
# Rscript 3b_stats_R/scripts_r/30_kruskal_extended.R
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
message("=== 30_kruskal_extended.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# CHARGEMENT DONNÉES SUPPLÉMENTAIRES
# ---------------------------------------------------------------------------

df_work <- df_clean

scene_csv <- file.path(dirname(BASE), "2d_vision", "keyframes", "results", "scene_detection.csv")
if (file.exists(scene_csv)) {
  scene_data <- tryCatch(read.csv(scene_csv, stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(scene_data) && nrow(scene_data) > 0) {
    scene_data$message_id <- as.integer(scene_data$message_id)
    scene_data <- scene_data[, c("message_id", "cuts_per_minute"), drop = FALSE]
    names(scene_data)[names(scene_data) == "cuts_per_minute"] <- "scene_cuts_per_min_csv"
    df_work <- merge(df_work, scene_data, by = "message_id", all.x = TRUE)
    message("  Scene CSV chargé : ", sum(!is.na(df_work$scene_cuts_per_min_csv)), " vidéos")
  }
}

# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------

compute_kruskal_dunn_ext <- function(data, metrics, p_adj_method = "BH") {
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
    d <- if (m == "duree_sec") {
      df_kw %>% filter(media_type == "video", duree_sec > 0)
    } else if (m == "scene_cuts_per_min_csv") {
      df_kw %>% filter(media_type == "video")
    } else {
      df_kw
    }
    d %>% filter(!is.na(.data[[m]]))
  }
  .ok <- function(d) nrow(d) >= 6 && n_distinct(d$phase) >= 2

  kruskal_res <- lapply(metrics, function(m) {
    if (!m %in% names(df_kw)) return(NULL)
    d <- .prep(m); if (!.ok(d)) return(NULL)
    tryCatch(rstatix::kruskal_test(d, as.formula(paste(m, "~ phase"))) %>% mutate(metric = m),
             error = function(e) NULL)
  })

  dunn_res <- lapply(metrics, function(m) {
    if (!m %in% names(df_kw)) return(NULL)
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

plot_kruskal_boxplots_ext <- function(data, metrics, ncol = 3,
                                      titre = "Distribution par phase — étendu (Kruskal-Wallis)") {
  if (!requireNamespace("rstatix", quietly = TRUE))
    stop("Package 'rstatix' requis.")
  library(rstatix)

  df_base <- data %>%
    filter(!is.na(phase)) %>%
    mutate(
      phase         = factor(phase, levels = c(1, 2, 3), labels = c("P1", "P2", "P3")),
      caption_nchar = stri_length(coalesce(as.character(legende), ""))
    )

  metrics <- metrics[metrics %in% names(df_base)]

  .prep <- function(m) {
    d <- if (m == "duree_sec") {
      df_base %>% filter(media_type == "video", duree_sec > 0)
    } else if (m == "scene_cuts_per_min_csv") {
      df_base %>% filter(media_type == "video")
    } else {
      df_base
    }
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
    duree_sec = "Durée vidéo (s)", caption_nchar = "Longueur legende",
    parole_ratio = "Speech ratio", n_reaction_types = "Nb types réactions",
    scene_cuts_per_min_csv = "Coupes/min (scène)", visages_magyar_ratio = "Ratio Magyar (visage)"
  )

  df_long <- bind_rows(lapply(metrics, function(m) {
    .prep(m) %>% transmute(phase, value = .data[[m]], metric = m)
  })) %>%
    left_join(kw_pvals, by = "metric") %>%
    mutate(
      metric_label = ifelse(metric %in% names(labels_metric), labels_metric[metric], metric),
      facet_label  = paste0(metric_label, "\n", pval_label),
      facet_label  = factor(facet_label,
        levels = paste0(
          ifelse(metrics %in% names(labels_metric), labels_metric[metrics], metrics),
          "\n",
          kw_pvals$pval_label[match(metrics, kw_pvals$metric)]
        )
      )
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
# RENDER
# ---------------------------------------------------------------------------

metrics_ext <- c("vues", "transferts", "reactions", "duree_sec", "caption_nchar",
                 "parole_ratio", "n_reaction_types")

if ("scene_cuts_per_min_csv" %in% names(df_work)) metrics_ext <- c(metrics_ext, "scene_cuts_per_min_csv")
if ("visages_magyar_ratio"     %in% names(df_work)) metrics_ext <- c(metrics_ext, "visages_magyar_ratio")

message("Métriques testées : ", paste(metrics_ext, collapse = ", "))

kw_results <- compute_kruskal_dunn_ext(df_work, metrics = metrics_ext)

write.csv(kw_results$kruskal, file.path(OUT, "30_kruskal_extended.csv"), row.names = FALSE)
message("\u2714 Sauvé: 30_kruskal_extended.csv")
write.csv(kw_results$dunn, file.path(OUT, "30_dunn_extended.csv"), row.names = FALSE)
message("\u2714 Sauvé: 30_dunn_extended.csv")

save_plot(
  plot_kruskal_boxplots_ext(df_work, metrics = metrics_ext, ncol = 3),
  file.path(OUT, "30_kruskal_extended_boxplots.png"),
  format = "wide_16_9", width = 16, dpi = 600
)

message("=== Terminé : 1 PNG + 2 CSV dans ", OUT, " ===")
