# 18_heatmap_horaire.R — Heatmap heure × jour de posting (global + par année/trimestre)
# Produit : 4_data_et_viz/18_heatmap_hour_global.png, 19_heatmap_annee.png, 21_heatmap_trim.png
# Rscript 3b_stats_R/scripts_r/18_heatmap_horaire.R

this_file <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  if (length(f) > 0 && nzchar(f)) return(normalizePath(f, mustWork = FALSE))
  for (env in rev(sys.frames()))
    if (!is.null(env$ofile)) return(normalizePath(env$ofile, mustWork = FALSE))
  stop("Lancer via Rscript ou source(). Ex : source('3b_stats_R/scripts_r/18_heatmap_horaire.R')")
})
BASE <- dirname(dirname(this_file))
source(file.path(BASE, "r_source.R"))
OUT  <- file.path(BASE, "..", "4_data_et_viz")
dir.create(OUT, showWarnings = FALSE)
message("=== 18_heatmap_horaire.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------

plot_heatmap_posting_hour <- function(data,
                                      titre = "Heure de publication",
                                      tz = "UTC") {
  jours_lvls <- tools::toTitleCase(gsub("\\.", "", levels(
    wday(seq.Date(as.Date("2024-01-01"), by = "day", length.out = 7),
         label = TRUE, abbr = TRUE, week_start = 1)
  )))

  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      date  = with_tz(date, tzone = tz),
      heure = hour(date),
      jour  = tools::toTitleCase(gsub("\\.", "", as.character(
                wday(date, label = TRUE, abbr = TRUE, week_start = 1))))
    ) %>%
    count(jour, heure, name = "n") %>%
    complete(
      jour  = jours_lvls,
      heure = 0:23,
      fill  = list(n = 0)
    ) %>%
    mutate(
      jour  = factor(jour, levels = jours_lvls),
      heure = factor(heure, levels = 0:23)
    )

  ggplot(df, aes(x = heure, y = jour, fill = n)) +
    geom_tile(color = "white", linewidth = 0.2) +
    scale_fill_gradient(low = PAL_SEQ_LOW, high = PAL_SEQ_HIGH) +
    labs(
      title    = titre,
      subtitle = "Nombre de publications par jour et par heure",
      x = "Heure (0–23)", y = NULL, fill = "Posts"
    )
}

plot_heatmap_posting_hour_by_period <- function(data,
                                                period = c("year", "quarter"),
                                                tz = "Europe/Paris",
                                                normalize = c("none", "share"),
                                                ncol = 3,
                                                cap_q = 0.98,
                                                x_angle = 0,
                                                stagger_x = FALSE,
                                                fill_label = NULL,
                                                fill_limits = NULL,
                                                source_caption = NULL,
                                                titre = NULL) {
  period    <- match.arg(period)
  normalize <- match.arg(normalize)

  # Niveaux des jours sans point (locale FR : "lun." → "lun"), semaine commence lundi.
  jours_levels <- tools::toTitleCase(gsub("\\.", "", levels(
    wday(seq.Date(as.Date("2024-01-01"), by = "day", length.out = 7),
         label = TRUE, abbr = TRUE, week_start = 1)
  )))

  df <- data %>%
    filter(!is.na(date)) %>%
    mutate(
      date  = with_tz(date, tzone = tz),
      heure = hour(date),
      jour  = tools::toTitleCase(gsub("\\.", "", as.character(
                wday(date, label = TRUE, abbr = TRUE, week_start = 1)))),
      periode = case_when(
        period == "year"    ~ as.character(year(date)),
        period == "quarter" ~ paste0(year(date), " Q", quarter(date))
      )
    ) %>%
    count(periode, jour, heure, name = "n") %>%
    complete(
      periode,
      jour  = jours_levels,
      heure = 0:23,
      fill  = list(n = 0)
    ) %>%
    mutate(
      jour  = factor(jour, levels = jours_levels),
      heure = factor(heure, levels = 0:23)
    )

  # Ordre chronologique pour les trimestres ("YYYY QN" trie correctement).
  if (period == "quarter") {
    df <- df %>%
      mutate(periode = factor(periode, levels = sort(unique(as.character(periode)))))
  }

  if (normalize == "share") {
    df <- df %>%
      group_by(periode) %>%
      mutate(value = ifelse(sum(n) == 0, 0, n / sum(n))) %>%
      ungroup()
    fill_lab <- "Part"
    subtitle  <- "Part des posts dans la période (chaque heatmap somme à 100 %)"
  } else {
    df       <- df %>% mutate(value = n)
    fill_lab <- if (!is.null(fill_label)) fill_label else "Posts"
    subtitle  <- "Nombre de publications par créneau (jour × heure)"
  }

  cap_q   <- max(0, min(1, cap_q))
  cap_val <- suppressWarnings(quantile(df$value, probs = cap_q, na.rm = TRUE))
  if (is.finite(cap_val) && cap_q < 1) {
    df <- df %>% mutate(value_plot = pmin(value, cap_val))
  } else {
    df <- df %>% mutate(value_plot = value)
  }

  if (is.null(titre)) {
    titre <- paste0("Heatmap d'heure de posting — par ",
                    ifelse(period == "year", "année", "trimestre"))
  }

  fill_scale <- if (normalize == "share")
    scale_fill_gradient(low = PAL_SEQ_LOW, high = PAL_SEQ_HIGH,
                        labels = percent_format(accuracy = 1),
                        limits = fill_limits, oob = scales::squish)
  else
    scale_fill_gradient(low = PAL_SEQ_LOW, high = PAL_SEQ_HIGH,
                        limits = fill_limits, oob = scales::squish)

  x_hjust <- if (x_angle >= 45) 1 else 0.5
  x_vjust <- if (x_angle >= 45) 1 else 0.5

  ggplot(df, aes(x = heure, y = jour, fill = value_plot)) +
    geom_tile(color = "white", linewidth = 0.2) +
    fill_scale +
    scale_x_discrete(labels = if (stagger_x) {
      function(x) { h <- as.integer(x); ifelse(h %% 2 == 0, paste0(h, "h"), paste0("\n", h, "h")) }
    } else {
      function(x) paste0(as.integer(x), "h")
    }) +
    facet_wrap(~periode, ncol = ncol) +
    labs(title = titre, subtitle = subtitle,
         x = NULL, y = NULL, fill = fill_lab,
         legende = source_caption) +
    theme_madyar_facet() +
    theme(
      axis.ticks.x = element_line(colour = "grey60", linewidth = 0.3),
      axis.text.x  = element_text(angle = x_angle, hjust = x_hjust,
                                  vjust = x_vjust, size = 8,
                                  margin = margin(t = 5))
    )
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_heatmap_posting_hour(df_clean, tz = "Europe/Paris"),
  file.path(OUT, "18_heatmap_hour_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)
save_plot(
  plot_heatmap_posting_hour_by_period(df_clean, period = "year",    normalize = "none",  ncol = 3),
  file.path(OUT, "19_heatmap_annee.png"),
  format = "wide_16_9", width = 14, dpi = 600
)
save_plot(
  plot_heatmap_posting_hour_by_period(
    df_clean, period = "quarter", normalize = "none", ncol = 4,
    cap_q          = 1,
    x_angle        = 45,
    fill_label     = "nb de posts par jour",
    fill_limits    = c(0, 15),
    titre          = "Carte de chaleur des heures de publication, par trimestre",
    source_caption = "Source : champ « date » (horodatage Telegram, tz Europe/Paris) — corpus @robert_magyar, 2022–2025"
  ),
  file.path(OUT, "21_heatmap_trim.png"),
  format = "wide_16_9", width = 14, dpi = 600
)

message("=== Terminé : 3 exports dans ", OUT, " ===")
