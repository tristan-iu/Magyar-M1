# 36_overlay_posts_duree_avg.R — Overlay fréquence de posting × durée moyenne
# Produit : 4_data_et_viz/36a_overlay_posts_duree_avg_global.png, 36c_overlay_posts_duree_avg_P3.png
# Rscript 3b_stats_R/scripts_r/36_overlay_posts_duree_avg.R

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
message("=== 36_overlay_posts_duree_avg.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# OUTLIER — msg 889 (30 juin 2024, 4991s = 83min, docu "Очі та Жало")
# ---------------------------------------------------------------------------
OUTLIER_IDS <- c(889L)
df_clean <- df_clean %>% filter(!message_id %in% OUTLIER_IDS)
message("Outlier(s) retirés : ", paste(OUTLIER_IDS, collapse = ", "),
        " — ", nrow(df_clean), " messages restants")

# ---------------------------------------------------------------------------
# DONNÉES
# ---------------------------------------------------------------------------

build_overlay_data <- function(data) {
  df_posts <- data %>%
    filter(!is.na(date)) %>%
    mutate(mois = as.Date(floor_date(date, "month"))) %>%
    count(mois, name = "n_posts") %>%
    complete(mois = seq(min(mois), max(mois), by = "month"),
             fill = list(n_posts = 0))

  df_duree <- data %>%
    filter(media_type == "video", !is.na(mois), !is.na(duree_sec), duree_sec > 0) %>%
    mutate(mois = as.Date(mois)) %>%
    group_by(mois) %>%
    summarise(duree_avg_sec = mean(duree_sec, na.rm = TRUE),
              n_videos = n(),
              .groups = "drop") %>%
    complete(mois = seq(min(mois), max(mois), by = "month"))

  full_join(df_posts, df_duree, by = "mois")
}

# ---------------------------------------------------------------------------
# PLOT GLOBAL — double axe Y
# ---------------------------------------------------------------------------

plot_overlay_global <- function(data,
                                titre = "Fréquence de posting & durée moyenne des vidéos") {
  df <- build_overlay_data(data)

  # Ratio pour le second axe : max(posts) ↔ max(durée)
  max_posts <- max(df$n_posts, na.rm = TRUE)
  max_duree <- max(df$duree_avg_sec, na.rm = TRUE)
  ratio     <- max_posts / max_duree

  pal <- c(
    "Publications / mois"   = unname(PAL_PHASE["1_Artisanal"]),
    "Durée moyenne (s)"     = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df, aes(x = mois)) +
    geom_phase_bands() +
    geom_col(aes(y = n_posts, fill = "Publications / mois"),
             alpha = 0.35, width = 25) +
    geom_line(aes(y = duree_avg_sec * ratio, colour = "Durée moyenne (s)"),
              linewidth = 0.9, na.rm = TRUE) +
    geom_point(aes(y = duree_avg_sec * ratio, colour = "Durée moyenne (s)"),
               size = 1.8, na.rm = TRUE) +
    scale_y_continuous(
      name = "Nombre de publications",
      expand = expansion(mult = c(0, 0.08)),
      sec.axis = sec_axis(~ . / ratio, name = "Durée moyenne (secondes)")
    ) +
    scale_x_mois(breaks = "3 months") +
    scale_fill_manual(values = pal) +
    scale_colour_manual(values = pal) +
    labs(
      title    = titre,
      subtitle = "Barres = publications/mois · Ligne = durée moyenne des vidéos (secondes)",
      x = NULL,
      caption  = NULL
    ) +
    theme(
      axis.title.y.right = element_text(colour = PAL_PHASE["2_Semi-pro"]),
      axis.text.y.right  = element_text(colour = PAL_PHASE["2_Semi-pro"]),
      axis.title.y.left  = element_text(colour = PAL_PHASE["1_Artisanal"]),
      axis.text.y.left   = element_text(colour = PAL_PHASE["1_Artisanal"])
    )
}

# ---------------------------------------------------------------------------
# PLOT PAR PHASE — facettes, indexé 0-100
# ---------------------------------------------------------------------------

plot_overlay_par_phase <- function(data,
                                   titre = "Fréquence & durée moyenne — par phase") {

  phase_labels <- c(
    "1" = "P1 Artisanal",
    "2" = "P2 Semi-pro",
    "3" = "P3 Institutionnel"
  )

  df_phase_all <- data %>%
    filter(!is.na(phase), !is.na(date)) %>%
    mutate(mois = as.Date(floor_date(date, "month")),
           phase_lbl = factor(phase_labels[as.character(phase)],
                              levels = phase_labels))

  # Posts par mois et par phase
  df_posts <- df_phase_all %>%
    count(phase_lbl, mois, name = "n_posts") %>%
    group_by(phase_lbl) %>%
    complete(mois = seq(min(mois), max(mois), by = "month"),
             fill = list(n_posts = 0)) %>%
    ungroup()

  # Durée moyenne par mois et par phase
  df_duree <- df_phase_all %>%
    filter(media_type == "video", !is.na(duree_sec), duree_sec > 0) %>%
    group_by(phase_lbl, mois) %>%
    summarise(duree_avg_sec = mean(duree_sec, na.rm = TRUE),
              .groups = "drop")

  # Jointure + indexation par phase
  df <- full_join(df_posts, df_duree, by = c("phase_lbl", "mois")) %>%
    group_by(phase_lbl) %>%
    mutate(
      posts_idx = to_index_100(n_posts),
      duree_idx = to_index_100(duree_avg_sec)
    ) %>%
    ungroup()

  # Pivot long
  df_long <- df %>%
    select(phase_lbl, mois, posts_idx, duree_idx) %>%
    pivot_longer(cols = c(posts_idx, duree_idx),
                 names_to = "serie", values_to = "index_100") %>%
    mutate(serie = recode(serie,
      posts_idx = "Posts / mois",
      duree_idx = "Durée moyenne vidéo"
    ))

  pal <- c(
    "Posts / mois"          = unname(PAL_PHASE["1_Artisanal"]),
    "Durée moyenne vidéo" = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df_long, aes(x = mois, y = index_100, colour = serie)) +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 1.5, na.rm = TRUE) +
    facet_wrap(~phase_lbl, scales = "free_x", nrow = 1) +
    scale_colour_manual(values = pal) +
    scale_x_mois(breaks = "2 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = titre,
      subtitle = "Indice 0–100 calculé indépendamment par phase et par série",
      x = NULL, y = "Indice (0–100)",
      caption  = paste0("Source : messages_clean.jsonl (", nrow(data),
                        " messages). Indexation min-max par phase.")
    ) +
    theme_madyar_facet() +
    theme(legend.position = "bottom")
}

# ---------------------------------------------------------------------------
# PLOT P3 DÉDIÉ — double axe Y, phase 3 seule
# ---------------------------------------------------------------------------

plot_overlay_p3 <- function(data,
                            titre = "P3 Institutionnel — fréquence & durée moyenne") {
  data_p3 <- data %>% filter(phase == 3L)
  df <- build_overlay_data(data_p3)

  max_posts <- max(df$n_posts, na.rm = TRUE)
  max_duree <- max(df$duree_avg_sec, na.rm = TRUE)
  ratio     <- max_posts / max_duree

  pal <- c(
    "Posts / mois"      = unname(PAL_PHASE["1_Artisanal"]),
    "Durée moyenne (s)" = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df, aes(x = mois)) +
    geom_col(aes(y = n_posts, fill = "Posts / mois"),
             alpha = 0.35, width = 25) +
    geom_line(aes(y = duree_avg_sec * ratio, colour = "Durée moyenne (s)"),
              linewidth = 0.9, na.rm = TRUE) +
    geom_point(aes(y = duree_avg_sec * ratio, colour = "Durée moyenne (s)"),
               size = 2.2, na.rm = TRUE) +
    scale_y_continuous(
      name = "Nombre de posts",
      expand = expansion(mult = c(0, 0.08)),
      sec.axis = sec_axis(~ . / ratio, name = "Durée moyenne (secondes)")
    ) +
    scale_x_mois(breaks = "1 month") +
    scale_fill_manual(values = pal) +
    scale_colour_manual(values = pal) +
    labs(
      title    = titre,
      subtitle = "Oct. 2024 – sept. 2025 — barres = posts, ligne = durée moyenne vidéo",
      x = NULL,
      caption  = "Hors outlier msg 889 (83 min). Ruptures PELT/MBIC."
    ) +
    theme(
      axis.title.y.right = element_text(colour = PAL_PHASE["2_Semi-pro"]),
      axis.text.y.right  = element_text(colour = PAL_PHASE["2_Semi-pro"]),
      axis.title.y.left  = element_text(colour = PAL_PHASE["1_Artisanal"]),
      axis.text.y.left   = element_text(colour = PAL_PHASE["1_Artisanal"])
    )
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

# -- Changepoint data (global) --
df_cpt_posts <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  count(mois) %>%
  complete(mois = mois_seq_df(df_clean), fill = list(n = 0))
df_cpt_posts <- .normalize_last_month_posts(df_cpt_posts, "n", LAST_OBS)$df

df_cpt_duree <- df_clean %>%
  filter(media_type == "video", !is.na(date), !is.na(duree_sec), duree_sec > 0) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  group_by(mois) %>%
  summarise(d = mean(duree_sec), .groups = "drop") %>%
  complete(mois = mois_seq_df(df_clean))

cpts_posts_global <- compute_cpts(df_cpt_posts$n,  df_cpt_posts$mois)
cpts_duree_global <- compute_cpts(df_cpt_duree$d, df_cpt_duree$mois)
message("CPT posts global : ", paste(format(cpts_posts_global), collapse = ", "))
message("CPT durée global : ", paste(format(cpts_duree_global), collapse = ", "))

# 36a — Global avec CPT
p_global <- plot_overlay_global(df_clean,
  "Fréquences de publication et durée moyenne des vidéos")
p_global <- add_cpt_lines(p_global, cpts_posts_global, color = PAL_PHASE["1_Artisanal"])
p_global <- add_cpt_lines(p_global, cpts_duree_global, color = PAL_PHASE["2_Semi-pro"])
save_plot(p_global,
  file.path(OUT, "36a_overlay_posts_duree_avg_global.png"),
  format = "wide_16_9", width = 10, dpi = 600)

# -- Changepoint data (P3) --
df_p3 <- df_clean %>% filter(phase == 3L)
df_cpt_posts_p3 <- df_p3 %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  count(mois) %>%
  complete(mois = seq(min(mois), max(mois), by = "month"), fill = list(n = 0))

df_cpt_duree_p3 <- df_p3 %>%
  filter(media_type == "video", !is.na(date), !is.na(duree_sec), duree_sec > 0) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  group_by(mois) %>%
  summarise(d = mean(duree_sec), .groups = "drop") %>%
  complete(mois = seq(min(mois), max(mois), by = "month"))

cpts_posts_p3 <- compute_cpts(df_cpt_posts_p3$n,  df_cpt_posts_p3$mois)
cpts_duree_p3 <- compute_cpts(df_cpt_duree_p3$d, df_cpt_duree_p3$mois)
message("CPT posts P3 : ", paste(format(cpts_posts_p3), collapse = ", "))
message("CPT durée P3 : ", paste(format(cpts_duree_p3), collapse = ", "))

# 36c — P3 dédié avec CPT
p_p3 <- plot_overlay_p3(df_clean)
p_p3 <- add_cpt_lines(p_p3, cpts_posts_p3, color = PAL_PHASE["1_Artisanal"])
p_p3 <- add_cpt_lines(p_p3, cpts_duree_p3, color = PAL_PHASE["2_Semi-pro"])
save_plot(p_p3,
  file.path(OUT, "36c_overlay_posts_duree_avg_P3.png"),
  format = "wide_16_9", width = 10, dpi = 600)

message("=== Terminé : 2 exports dans ", OUT, " ===")
