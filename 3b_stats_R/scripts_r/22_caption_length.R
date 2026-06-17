# 22_caption_length.R — Taille des captions (médiane + IQR, ribbon) — global + phases + cpt
# Produit : 4_data_et_viz/22_caption_length_global{,_cpt}.png, 23-25_caption_length_p{1,2,3}.png
# Rscript 3b_stats_R/scripts_r/22_caption_length.R

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
message("=== 22_caption_length.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTION
# ---------------------------------------------------------------------------

plot_caption_length_ribbon_mois <- function(data,
                                            titre = "Taille des captions (mensuel)",
                                            tz = "Europe/Paris",
                                            drop_empty = TRUE,
                                            phase_lines = TRUE) {
  df_mois <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      date         = with_tz(date, tzone = tz),
      mois         = as.Date(floor_date(date, "month")),
      caption_txt  = dplyr::coalesce(as.character(legende), ""),
      caption_nchar = stri_length(caption_txt)
    ) %>%
    { if (drop_empty) dplyr::filter(., caption_nchar > 0) else . } %>%
    group_by(mois) %>%
    summarise(
      n_msgs = dplyr::n(),
      p25    = as.numeric(quantile(caption_nchar, 0.25, na.rm = TRUE, names = FALSE)),
      p50    = as.numeric(quantile(caption_nchar, 0.50, na.rm = TRUE, names = FALSE)),
      p75    = as.numeric(quantile(caption_nchar, 0.75, na.rm = TRUE, names = FALSE)),
      .groups = "drop"
    ) %>%
    complete(
      mois = seq(min(mois), max(mois), by = "month"),
      fill = list(n_msgs = 0, p25 = NA_real_, p50 = NA_real_, p75 = NA_real_)
    ) %>%
    arrange(mois)

  df_line <- df_mois %>% filter(!is.na(p50))
  df_rib  <- df_mois %>% filter(!is.na(p25), !is.na(p75))

  p <- ggplot() +
    geom_ribbon(data = df_rib, aes(x = mois, ymin = p25, ymax = p75),
                fill = unname(PAL_PHASE["1_Artisanal"]), alpha = 0.20) +
    geom_line(data = df_line, aes(x = mois, y = p50),
              linewidth = 0.9, colour = unname(PAL_PHASE["1_Artisanal"])) +
    geom_point(data = df_line, aes(x = mois, y = p50),
               size = 1.8, colour = unname(PAL_PHASE["1_Artisanal"])) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(
      title    = titre,
      subtitle = paste0("Médiane + IQR (P25–P75)",
                        if (drop_empty) " — captions vides exclues" else ""),
      x = NULL, y = "Nombre de caractères",
      caption  = "Source : messages_clean.jsonl. Taille = stri_length(). Ruban = P25–P75."
    )

  if (phase_lines) p <- p + geom_phase_lines()
  p
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_caption_length_ribbon_mois(df_clean, titre = "Taille des captions — période complète"),
  file.path(OUT, "22_caption_length_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_caption_length_ribbon_mois(df_phase(1), titre = "Taille des captions — phase 1", phase_lines = FALSE),
  file.path(OUT, "23_caption_length_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_caption_length_ribbon_mois(df_phase(2), titre = "Taille des captions — phase 2", phase_lines = FALSE),
  file.path(OUT, "24_caption_length_p2.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_caption_length_ribbon_mois(df_phase(3), titre = "Taille des captions — phase 3", phase_lines = FALSE),
  file.path(OUT, "25_caption_length_p3.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variante avec changepoints
df_c <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(
    mois  = as.Date(floor_date(date, "month")),
    nchar = stri_length(coalesce(as.character(legende), ""))
  ) %>%
  filter(nchar > 0) %>%
  group_by(mois) %>% summarise(med = median(nchar), .groups = "drop") %>%
  complete(mois = mois_seq_df(df_clean))

save_plot(
  add_cpt_lines(
    plot_caption_length_ribbon_mois(df_clean, "Longueur captions — ruptures détectées"),
    compute_cpts(df_c$med, df_c$mois)
  ),
  file.path(OUT, "22_caption_length_global_cpt.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

message("=== Terminé : 5 exports dans ", OUT, " ===")
