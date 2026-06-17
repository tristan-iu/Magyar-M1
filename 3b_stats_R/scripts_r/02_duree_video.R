# 02_duree_video.R — Durée médiane des vidéos + overlay posts/durée + changepoints
# Produit : 4_data_et_viz/02_duree_video_mois{,_cpt}.png, 03_overlay_posts_duree{,_cpt}.png
# Rscript 3b_stats_R/scripts_r/02_duree_video.R

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
message("=== 02_duree_video.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------

plot_duree_video_mois <- function(data,
                                  titre = "Durée médiane des vidéos par mois") {
  df <- data %>%
    filter(media_type == "video", !is.na(mois), !is.na(duree_sec), duree_sec > 0) %>%
    mutate(mois = as.Date(mois)) %>%
    group_by(mois) %>%
    summarise(
      duree_med_sec = median(duree_sec, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    complete(mois = seq(min(mois), max(mois), by = "month"))

  ggplot(df, aes(x = mois, y = duree_med_sec)) +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.7, na.rm = TRUE) +
    geom_point(colour = PAL_PHASE["2_Semi-pro"], size = 1.8, na.rm = TRUE) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(suffix = " s"),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle — robuste aux outliers (quelques vidéos > 1h)",
      x = NULL, y = "Durée médiane (secondes)",
      caption  = "Source : messages_clean.jsonl (1 365 messages). Vidéos uniquement (duree_sec > 0)."
    )
}

plot_overlay_posts_duree <- function(data,
                                     titre = "Activité & durée vidéo (mensuel)") {
  df_posts <- data %>%
    filter(!is.na(mois)) %>%
    mutate(mois = as.Date(mois)) %>%
    count(mois, name = "n_posts") %>%
    complete(mois = seq(min(mois), max(mois), by = "month"), fill = list(n_posts = 0))

  df_video <- data %>%
    filter(media_type == "video", !is.na(mois), !is.na(duree_sec), duree_sec > 0) %>%
    mutate(mois = as.Date(mois)) %>%
    group_by(mois) %>%
    summarise(duree_med_sec = median(duree_sec, na.rm = TRUE), .groups = "drop") %>%
    complete(mois = seq(min(mois), max(mois), by = "month"))

  pal_overlay <- c(
    "Nombre de posts"     = unname(PAL_PHASE["1_Artisanal"]),
    "Durée médiane vidéo" = unname(PAL_PHASE["2_Semi-pro"])
  )

  full_join(df_posts, df_video, by = "mois") %>%
    mutate(
      posts_idx = to_index_100(n_posts),
      duree_idx = to_index_100(duree_med_sec)
    ) %>%
    select(mois, posts_idx, duree_idx) %>%
    pivot_longer(-mois, names_to = "serie", values_to = "index_100") %>%
    mutate(serie = recode(serie,
      posts_idx = "Nombre de posts",
      duree_idx = "Durée médiane vidéo"
    )) %>%
    ggplot(aes(x = mois, y = index_100, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 1.8, na.rm = TRUE) +
    scale_colour_manual(values = pal_overlay) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = titre,
      subtitle = "Deux séries indexées séparément (0 = min, 100 = max de chaque série)",
      x = NULL, y = "Indice (0–100)",
      caption  = "Source : messages_clean.jsonl. Indice min-max calculé indépendamment pour chaque série."
    )
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_duree_video_mois(df_clean, "Durée médiane des vidéos par mois"),
  file.path(OUT, "02_duree_video_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

save_plot(
  plot_overlay_posts_duree(df_clean, "Activité & durée vidéo (mensuel)"),
  file.path(OUT, "03_overlay_posts_duree.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variantes avec changepoints
df_p <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  count(mois) %>%
  complete(mois = mois_seq_df(df_clean), fill = list(n = 0))

df_v <- df_clean %>%
  filter(media_type == "video", !is.na(date), !is.na(duree_sec), duree_sec > 0) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  group_by(mois) %>% summarise(duree_med = median(duree_sec), .groups = "drop") %>%
  complete(mois = mois_seq_df(df_clean))

save_plot(
  add_cpt_lines(
    plot_duree_video_mois(df_clean, "Durée méd. vidéos — ruptures détectées"),
    compute_cpts(df_v$duree_med, df_v$mois),
    color = PAL_PHASE["3_Institutionnel"]
  ),
  file.path(OUT, "02_duree_video_mois_cpt.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

p03 <- plot_overlay_posts_duree(df_clean, "Activité & durée vidéo — ruptures détectées")
p03 <- add_cpt_lines(p03, compute_cpts(df_p$n,         df_p$mois),  color = PAL_PHASE["1_Artisanal"])
p03 <- add_cpt_lines(p03, compute_cpts(df_v$duree_med, df_v$mois),  color = PAL_PHASE["2_Semi-pro"])
save_plot(p03, file.path(OUT, "03_overlay_posts_duree_cpt.png"),
          format = "wide_16_9", width = 10, dpi = 600)

message("=== Terminé : 4 exports dans ", OUT, " ===")
