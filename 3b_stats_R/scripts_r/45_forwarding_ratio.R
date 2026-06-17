# 45_forwarding_ratio.R — Viralité : transferts/vues par phase
# Produit : 4_data_et_viz/45a_fwd_ratio_mois{,_cpt}.png, 45b_fwd_ratio_phase_boxplot.png,
#           45c_viralite_engagement_overlay.png
# Rscript 3b_stats_R/scripts_r/45_forwarding_ratio.R

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
message("=== 45_forwarding_ratio.R — ", nrow(df_clean), " messages chargés ===")

# Forwards/vues ratio — exclure les vues = 0
df_fwd <- df_clean |>
  filter(!is.na(mois), vues > 0) |>
  mutate(
    mois = as.Date(mois),
    fwd_ratio = transferts / vues,
    react_ratio = reactions / vues,
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  )

n_posts <- nrow(df_fwd)
message("  Posts avec vues > 0 : ", n_posts)

# ---------------------------------------------------------------------------
# 45a — Forwarding ratio médian par mois
# ---------------------------------------------------------------------------

df_mois_fwd <- df_fwd |>
  group_by(mois) |>
  summarise(
    fwd_ratio_med  = median(fwd_ratio, na.rm = TRUE),
    react_ratio_med = median(react_ratio, na.rm = TRUE),
    n_posts = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

plot_fwd_mois <- function(df, titre = "Ratio transferts/vues par mois") {
  ggplot(df, aes(x = mois, y = fwd_ratio_med * 1000)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(aes(size = n_posts), colour = PAL_PHASE["2_Semi-pro"],
               alpha = 0.85, na.rm = TRUE) +
    scale_size_continuous(range = c(1.5, 5), name = "Nb posts") +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      expand = expansion(mult = c(0.02, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Médiane mensuelle (transferts / vues × 1000) — proxy de viralité",
      x = NULL, y = "Forwards pour 1 000 vues (médiane)",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", n_posts,
        " posts avec vues > 0."
      )
    )
}

save_plot(
  plot_fwd_mois(df_mois_fwd),
  file.path(OUT, "45a_fwd_ratio_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variante avec changepoints
cpts_fwd <- compute_cpts(df_mois_fwd$fwd_ratio_med, df_mois_fwd$mois)
if (length(cpts_fwd) > 0) {
  save_plot(
    add_cpt_lines(
      plot_fwd_mois(df_mois_fwd, "Forwarding ratio — ruptures détectées"),
      cpts_fwd, color = PAL_PHASE["3_Institutionnel"]
    ),
    file.path(OUT, "45a_fwd_ratio_mois_cpt.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# ---------------------------------------------------------------------------
# 45b — Boxplot forwarding ratio par phase
# ---------------------------------------------------------------------------

df_fwd_ph <- df_fwd |> filter(!is.na(phase_lbl))

med_fwd <- df_fwd_ph |>
  group_by(phase_lbl) |>
  summarise(
    med = median(fwd_ratio * 1000, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

plot_box_fwd <- function() {
  ggplot(df_fwd_ph, aes(x = phase_lbl, y = fwd_ratio * 1000, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_fwd,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %.1f", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    coord_cartesian(ylim = c(0, quantile(df_fwd_ph$fwd_ratio * 1000, 0.95, na.rm = TRUE) * 1.3)) +
    labs(
      title    = "Viralité par phase",
      subtitle = paste0("Forwards pour 1 000 vues — ", sum(med_fwd$n), " posts"),
      x = NULL, y = "Forwards / 1 000 vues",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d", LBL_PHASE_SHORT[as.character(med_fwd$phase_lbl)],
              med_fwd$n), collapse = " | "),
        ". Tronqué au 95e percentile."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_box_fwd(),
  file.path(OUT, "45b_fwd_ratio_phase_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 45c — Overlay : forwarding ratio vs reaction ratio (indexés)
# ---------------------------------------------------------------------------

df_mois_ov <- df_mois_fwd |>
  filter(!is.na(fwd_ratio_med), !is.na(react_ratio_med)) |>
  mutate(
    fwd_idx   = to_index_100(fwd_ratio_med),
    react_idx = to_index_100(react_ratio_med)
  )

if (nrow(df_mois_ov) >= 3) {
  pal_eng <- c(
    "Forwarding (viralité)"   = unname(PAL_PHASE["2_Semi-pro"]),
    "Réactions (engagement)"  = unname(PAL_PHASE["1_Artisanal"])
  )

  p_ov <- df_mois_ov |>
    select(mois, fwd_idx, react_idx) |>
    pivot_longer(-mois, names_to = "serie", values_to = "idx") |>
    mutate(serie = recode(serie,
      fwd_idx   = "Forwarding (viralité)",
      react_idx = "Réactions (engagement)"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 1.8, na.rm = TRUE) +
    scale_colour_manual(values = pal_eng) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "Viralité vs Engagement (indexés)",
      subtitle = "Indices 0–100 — transferts/vues vs réactions/vues",
      x = NULL, y = "Indice (0 = min, 100 = max)",
      caption  = "Source : messages_clean.jsonl. Médiane mensuelle des ratios."
    )

  save_plot(
    p_ov,
    file.path(OUT, "45c_viralite_engagement_overlay.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# Résumé
message("\n=== Forwarding ratio par phase ===")
print(as.data.frame(med_fwd))
message("=== Terminé : exports dans ", OUT, " ===")
