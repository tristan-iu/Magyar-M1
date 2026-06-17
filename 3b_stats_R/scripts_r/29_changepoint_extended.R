# 29_changepoint_extended.R — Changepoint étendu (5 séries + parole_ratio, cuts/min)
# Produit : 4_data_et_viz/29_changepoint_extended.png, 29_ruptures_vs_phases.csv
# Rscript 3b_stats_R/scripts_r/29_changepoint_extended.R
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
message("=== 29_changepoint_extended.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# CHARGEMENT DONNÉES SUPPLÉMENTAIRES
# ---------------------------------------------------------------------------

scene_csv <- file.path(dirname(BASE), "2d_vision", "keyframes", "results", "scene_detection.csv")
has_scene <- FALSE
df_work   <- df_clean

if (file.exists(scene_csv)) {
  scene_data <- tryCatch(read.csv(scene_csv, stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(scene_data) && nrow(scene_data) > 0) {
    scene_data$message_id <- as.integer(scene_data$message_id)
    scene_data <- scene_data[, c("message_id", "cuts_per_minute"), drop = FALSE]
    df_work <- merge(df_work, scene_data, by = "message_id", all.x = TRUE)
    has_scene <- TRUE
    message("  Scene CSV chargé : ", nrow(scene_data), " lignes")
  }
}

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_changepoint_extended <- function(data, has_scene_data = FALSE,
                                      titre = "Détection de ruptures étendue (changepoint PELT)",
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

  serie_speech <- df_base %>%
    filter(media_type %in% c("video", "audio"), !is.na(parole_ratio), parole_ratio > 0) %>%
    group_by(mois) %>%
    summarise(value = median(parole_ratio, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = mois_seq) %>%
    mutate(serie = "speech_ratio_med")

  all_series <- list(
    serie_posts, serie_duree, serie_caption,
    serie_resolution, serie_orientation, serie_speech
  )

  labels_serie <- c(
    n_posts           = "Nb posts / mois",
    duree_med_sec     = "Durée méd. vidéo (s)",
    caption_nchar_med = "Longueur legende (méd., car.)",
    resolution_med    = "Résolution méd. (pixels)",
    pct_vertical      = "% format vertical",
    speech_ratio_med  = "Speech ratio méd."
  )

  if (has_scene_data && "cuts_per_minute" %in% names(df_base)) {
    serie_cuts <- df_base %>%
      filter(media_type == "video", !is.na(cuts_per_minute), cuts_per_minute >= 0) %>%
      group_by(mois) %>%
      summarise(value = median(cuts_per_minute, na.rm = TRUE), .groups = "drop") %>%
      complete(mois = mois_seq) %>%
      mutate(serie = "cuts_per_min_med")
    all_series <- c(all_series, list(serie_cuts))
    labels_serie <- c(labels_serie, cuts_per_min_med = "Coupes/min méd.")
  }

  df_long <- bind_rows(all_series) %>% arrange(serie, mois)

  ruptures <- df_long %>%
    group_by(serie) %>%
    summarise(rupture_date = list(compute_cpts(value, mois)), .groups = "drop") %>%
    unnest(cols = rupture_date)

  df_long  <- df_long  %>% mutate(serie = factor(serie, levels = names(labels_serie), labels = unname(labels_serie)))
  ruptures <- ruptures %>% mutate(serie = factor(serie, levels = names(labels_serie), labels = unname(labels_serie)))

  n_series <- length(labels_serie)

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
      subtitle = paste0("PELT, pénalité MBIC — ", n_series, " séries"),
      x = NULL, y = NULL,
      caption  = "Package changepoint (cpt.meanvar). Lignes rouges = ruptures détectées."
    ) +
    theme_madyar_facet()
}

# ---------------------------------------------------------------------------
# TABLEAU RUPTURES vs BORNES MANUELLES
# ---------------------------------------------------------------------------

export_ruptures_vs_phases <- function(data, has_scene_data = FALSE, tz = "Europe/Paris") {
  library(changepoint)

  df_base <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      date          = with_tz(date, tzone = tz),
      mois          = as.Date(floor_date(date, "month")),
      caption_nchar = stri_length(coalesce(as.character(legende), ""))
    )

  mois_seq <- seq(min(df_base$mois), max(df_base$mois), by = "month")

  series_list <- list(
    n_posts = df_base %>% count(mois, name = "value") %>%
      complete(mois = mois_seq, fill = list(value = 0)),
    duree_med_sec = df_base %>%
      filter(media_type == "video", !is.na(duree_sec), duree_sec > 0) %>%
      group_by(mois) %>% summarise(value = median(duree_sec), .groups = "drop") %>%
      complete(mois = mois_seq),
    caption_nchar_med = df_base %>%
      filter(stri_length(coalesce(as.character(legende), "")) > 0) %>%
      mutate(caption_nchar = stri_length(as.character(legende))) %>%
      group_by(mois) %>% summarise(value = median(caption_nchar), .groups = "drop") %>%
      complete(mois = mois_seq),
    resolution_med = df_base %>%
      filter(media_type %in% c("video", "photo"), !is.na(largeur), !is.na(hauteur)) %>%
      mutate(pixels = largeur * hauteur) %>%
      group_by(mois) %>% summarise(value = median(pixels), .groups = "drop") %>%
      complete(mois = mois_seq),
    pct_vertical = df_base %>%
      filter(media_type %in% c("video", "photo"), !is.na(orientation)) %>%
      mutate(is_vertical = tolower(orientation) == "vertical") %>%
      group_by(mois) %>% summarise(value = mean(is_vertical), .groups = "drop") %>%
      complete(mois = mois_seq),
    speech_ratio_med = df_base %>%
      filter(media_type %in% c("video", "audio"), !is.na(parole_ratio), parole_ratio > 0) %>%
      group_by(mois) %>% summarise(value = median(parole_ratio), .groups = "drop") %>%
      complete(mois = mois_seq)
  )

  series_list$n_posts <- .normalize_last_month_posts(series_list$n_posts, "value", LAST_OBS)$df

  if (has_scene_data && "cuts_per_minute" %in% names(df_base)) {
    series_list$cuts_per_min_med <- df_base %>%
      filter(media_type == "video", !is.na(cuts_per_minute), cuts_per_minute >= 0) %>%
      group_by(mois) %>% summarise(value = median(cuts_per_minute), .groups = "drop") %>%
      complete(mois = mois_seq)
  }

  rows <- lapply(names(series_list), function(nm) {
    s <- series_list[[nm]]
    cpts <- compute_cpts(s$value, s$mois)
    if (length(cpts) == 0) return(data.frame(serie = nm, rupture_date = NA_character_, stringsAsFactors = FALSE))
    data.frame(serie = nm, rupture_date = as.character(cpts), stringsAsFactors = FALSE)
  })

  ruptures_df <- do.call(rbind, rows)
  # Bornes sourcées depuis r_source.R (source unique = config.yaml)
  ruptures_df$borne_P1_P2 <- as.character(bornes$p2[1])
  ruptures_df$borne_P2_P3 <- as.character(bornes$p3[1])
  ruptures_df
}

# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------

p <- plot_changepoint_extended(df_work, has_scene_data = has_scene)
save_plot(p, file.path(OUT, "29_changepoint_extended.png"), format = "square", width = 10, dpi = 600)

rupt_table <- export_ruptures_vs_phases(df_work, has_scene_data = has_scene)
write.csv(rupt_table, file.path(OUT, "29_ruptures_vs_phases.csv"), row.names = FALSE)
message("\u2714 Sauvé: 29_ruptures_vs_phases.csv")

message("=== Terminé : 1 PNG + 1 CSV dans ", OUT, " ===")
