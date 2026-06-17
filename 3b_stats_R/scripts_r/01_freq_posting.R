# 01_freq_posting.R — Fréquence de posting mensuelle + changepoints
# Produit : 4_data_et_viz/01_freq_posting_global.png, 01_freq_posting_global_cpt.png
# Rscript 3b_stats_R/scripts_r/01_freq_posting.R

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
message("=== 01_freq_posting.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_posting_freq_mensuelle <- function(data, titre,
                                        normalize_last_month = TRUE,
                                        last_obs_date = LAST_OBS) {
  df_mois <- data %>%
    filter(!is.na(date)) %>%
    mutate(mois = as.Date(lubridate::floor_date(date, "month"))) %>%
    count(mois, name = "n_posts") %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      fill = list(n_posts = 0)
    )

  extrapol_mois <- NULL
  if (normalize_last_month && !is.null(last_obs_date)) {
    norm          <- .normalize_last_month_posts(df_mois, "n_posts", last_obs_date)
    df_mois       <- norm$df
    extrapol_mois <- norm$extrapol_mois
  }

  df_regular  <- if (!is.null(extrapol_mois)) dplyr::filter(df_mois, mois != extrapol_mois) else df_mois
  df_extrapol <- if (!is.null(extrapol_mois)) dplyr::filter(df_mois, mois == extrapol_mois) else df_mois[0, ]

  cap_note <- if (!is.null(extrapol_mois))
    "\n\u25c6 Dernier mois extrapolé : posts observés \u00d7 (jours du mois / jours couverts)."
  else ""

  p <- ggplot(df_mois, aes(x = mois, y = n_posts)) +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.7) +
    geom_point(data = df_regular, colour = PAL_PHASE["1_Artisanal"], size = 1.8) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(
      title    = titre,
      subtitle = "Nombre de posts par mois — @robert_magyar, sept. 2022 – sept. 2025",
      x = NULL, y = "Nombre de posts",
      caption  = paste0(
        "Source : messages_clean.jsonl (1 365 messages). ",
        "Mois manquants imputés à 0.", cap_note
      )
    )

  if (nrow(df_extrapol) > 0) {
    p <- p +
      geom_point(data = df_extrapol, aes(x = mois, y = n_posts),
                 colour = "#E67E22", size = 3.5, shape = 18) +
      annotate("text",
               x = df_extrapol$mois[1], y = df_extrapol$n_posts[1],
               label = "mois\nextrapolé", vjust = -0.6, hjust = 0.5,
               size = 3, colour = "#E67E22")
  }
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_posting_freq_mensuelle(df_clean, "Fréquence de posting (période complète)"),
  file.path(OUT, "01_freq_posting_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variante avec changepoints
df_m <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  count(mois) %>%
  complete(mois = mois_seq_df(df_clean), fill = list(n = 0))
df_m <- .normalize_last_month_posts(df_m, "n", LAST_OBS)$df

save_plot(
  add_cpt_lines(
    plot_posting_freq_mensuelle(df_clean, "Fréquence de posting — ruptures détectées"),
    compute_cpts(df_m$n, df_m$mois)
  ),
  file.path(OUT, "01_freq_posting_global_cpt.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

message("=== Terminé : 2 exports dans ", OUT, " ===")
