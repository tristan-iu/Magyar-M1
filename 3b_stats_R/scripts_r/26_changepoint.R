# 26_changepoint.R — Changepoint detection multi-séries (PELT, MBIC)
# Produit : 4_data_et_viz/26_changepoint_multiseries.png
# Rscript 3b_stats_R/scripts_r/26_changepoint.R
# Requiert : install.packages("changepoint")

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
message("=== 26_changepoint.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_changepoint_multiseries <- function(data,
                                         titre = "Détection de ruptures (changepoint PELT)",
                                         tz = "Europe/Paris") {
  if (!requireNamespace("changepoint", quietly = TRUE))
    stop("Package 'changepoint' requis : install.packages('changepoint')")
  library(changepoint)

  df_base <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      date          = with_tz(date, tzone = tz),
      mois          = as.Date(floor_date(date, "month")),
      caption_nchar = stri_length(coalesce(as.character(legende), ""))
    )

  mois_seq <- seq(min(df_base$mois), max(df_base$mois), by = "month")

  serie_posts <- df_base %>%
    count(mois, name = "value") %>%
    complete(mois = mois_seq, fill = list(value = 0)) %>%
    mutate(serie = "n_posts")
  serie_posts <- .normalize_last_month_posts(serie_posts, "value", LAST_OBS)$df

  serie_duree <- df_base %>%
    filter(media_type == "video", !is.na(duree_sec), duree_sec > 0) %>%
    group_by(mois) %>%
    summarise(value = median(duree_sec, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = mois_seq) %>%
    mutate(serie = "duree_med_sec")

  serie_caption <- df_base %>%
    filter(caption_nchar > 0) %>%
    group_by(mois) %>%
    summarise(value = median(caption_nchar, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = mois_seq) %>%
    mutate(serie = "caption_nchar_med")

  serie_resolution <- df_base %>%
    filter(media_type %in% c("video", "photo"),
           !is.na(largeur), !is.na(hauteur)) %>%
    mutate(pixels = largeur * hauteur) %>%
    group_by(mois) %>%
    summarise(value = median(pixels, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = mois_seq) %>%
    mutate(serie = "resolution_med")

  serie_orientation <- df_base %>%
    filter(media_type %in% c("video", "photo"), !is.na(orientation)) %>%
    mutate(is_vertical = tolower(orientation) == "vertical") %>%
    group_by(mois) %>%
    summarise(value = mean(is_vertical, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = mois_seq) %>%
    mutate(serie = "pct_vertical")

  df_long <- bind_rows(
    serie_posts, serie_duree, serie_caption,
    serie_resolution, serie_orientation
  ) %>%
    arrange(serie, mois)

  ruptures <- df_long %>%
    group_by(serie) %>%
    summarise(rupture_date = list(compute_cpts(value, mois)), .groups = "drop") %>%
    unnest(cols = rupture_date)

  labels_serie <- c(
    n_posts           = "Nb posts / mois",
    duree_med_sec     = "Durée méd. vidéo (s)",
    caption_nchar_med = "Longueur legende (méd., car.)",
    resolution_med    = "Résolution méd. (pixels)",
    pct_vertical      = "% format vertical"
  )

  df_long  <- df_long  %>% mutate(serie = factor(serie, levels = names(labels_serie), labels = unname(labels_serie)))
  ruptures <- ruptures %>% mutate(serie = factor(serie, levels = names(labels_serie), labels = unname(labels_serie)))

  ggplot(df_long, aes(x = mois, y = value)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.7, colour = "grey40", na.rm = TRUE) +
    geom_point(size = 1.4, colour = "grey40", na.rm = TRUE) +
    geom_vline(data = ruptures, aes(xintercept = as.numeric(rupture_date)),
               colour = "firebrick", linetype = "dashed", linewidth = 0.7) +
    facet_wrap(~serie, ncol = 1, scales = "free_y") +
    scale_x_mois(breaks = "6 months") +
    labs(
      title    = titre,
      subtitle = "PELT, pénalité MBIC (conservatrice, vise 2–3 ruptures/série)",
      x = NULL, y = NULL,
      caption  = "Package changepoint (cpt.meanvar). 5 séries : posting, durée méd., caption, résolution, orientation."
    ) +
    theme_madyar_facet()
}

# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------

save_plot(
  plot_changepoint_multiseries(df_clean),
  file.path(OUT, "26_changepoint_multiseries.png"),
  format = "square", width = 10, dpi = 600
)

message("=== Terminé : 1 export dans ", OUT, " ===")
