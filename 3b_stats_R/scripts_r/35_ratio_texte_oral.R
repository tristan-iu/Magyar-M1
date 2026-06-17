# 35_ratio_texte_oral.R — Rapport tokens legende (écrit) / tokens dialogue (oral)
# Produit : 4_data_et_viz/35a_tokens_stacked.png, 35b_pct_tokens.png,
#           35c_ratio_cap_dia{,_cpt}.png, 35d_ratio_par_phase{,_cpt}.png
# Rscript 3b_stats_R/scripts_r/35_ratio_texte_oral.R

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

LEX  <- normalizePath(file.path(BASE, "..", "4_data_et_viz", "lexico"))
message("=== 35_ratio_texte_oral.R ===")

# ---------------------------------------------------------------------------
# Chargement des lemmes (un token = une ligne)
# ---------------------------------------------------------------------------

cap_raw <- read.csv(file.path(LEX, "lemmes_caption.csv"),
                    stringsAsFactors = FALSE, encoding = "UTF-8")
dia_raw <- read.csv(file.path(LEX, "lemmes_dialogue.csv"),
                    stringsAsFactors = FALSE, encoding = "UTF-8")

message("  Tokens legende  : ", nrow(cap_raw))
message("  Tokens dialogue : ", nrow(dia_raw))

parse_lex_date <- function(x) {
  # Dates de la forme "2022-09-01T14:52:35"
  suppressWarnings(as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC"))
}

cap <- cap_raw |>
  mutate(
    date = parse_lex_date(date),
    mois = as.Date(floor_date(date, "month"))
  )

dia <- dia_raw |>
  mutate(
    date = parse_lex_date(date),
    mois = as.Date(floor_date(date, "month"))
  )

# ---------------------------------------------------------------------------
# Agrégation : tokens par message_id puis par mois
# ---------------------------------------------------------------------------

# Tokens par mois (somme de tous tokens de tous messages)
tokens_cap_mois <- cap |>
  filter(!is.na(mois)) |>
  count(mois, name = "tokens_caption")

tokens_dia_mois <- dia |>
  filter(!is.na(mois)) |>
  count(mois, name = "tokens_dialogue")

# Séquence complète de mois couvrant les deux corpus
all_mois <- seq(
  min(c(tokens_cap_mois$mois, tokens_dia_mois$mois), na.rm = TRUE),
  max(c(tokens_cap_mois$mois, tokens_dia_mois$mois), na.rm = TRUE),
  by = "month"
)

df_mois <- tibble(mois = all_mois) |>
  left_join(tokens_cap_mois, by = "mois") |>
  left_join(tokens_dia_mois, by = "mois") |>
  mutate(
    tokens_caption  = replace_na(tokens_caption, 0L),
    tokens_dialogue = replace_na(tokens_dialogue, 0L),
    total           = tokens_caption + tokens_dialogue,
    # Ratio legende/dialogue (legende pur / parole pure)
    # NA si les deux sont 0 ; on protège aussi div/0
    ratio_cap_dia   = if_else(
      tokens_dialogue > 0,
      tokens_caption / tokens_dialogue,
      NA_real_
    ),
    # Part relative du legende dans le total textuel
    pct_caption     = if_else(total > 0, tokens_caption / total * 100, NA_real_),
    pct_dialogue    = if_else(total > 0, tokens_dialogue / total * 100, NA_real_)
  )

message("  Mois couverts : ", nrow(df_mois),
        " (", min(all_mois), " → ", max(all_mois), ")")

# ---------------------------------------------------------------------------
# 35a — Volumes bruts empilés (legende vs dialogue)
# ---------------------------------------------------------------------------

plot_stacked_tokens <- function(df,
  titre = "Volume mensuel de tokens : texte écrit vs parole") {

  df_long <- df |>
    select(mois, tokens_caption, tokens_dialogue) |>
    pivot_longer(-mois, names_to = "source", values_to = "tokens") |>
    mutate(source = recode(source,
      tokens_caption  = "Caption (texte écrit)",
      tokens_dialogue = "Dialogue (transcription Whisper)"
    ))

  pal_src <- c(
    "Caption (texte écrit)"              = unname(PAL_PHASE["1_Artisanal"]),
    "Dialogue (transcription Whisper)"   = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df_long, aes(x = mois, y = tokens, fill = source)) +
    geom_phase_lines() +
    geom_area(alpha = 0.75, position = "stack") +
    scale_fill_manual(values = pal_src) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(big.mark = "\u202F"),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(
      title    = titre,
      subtitle = "Tokens lemmatisés (spaCy uk_core_news_trf) — empilés par mois",
      x = NULL, y = "Nombre de tokens",
      caption  = paste0(
        "Source : lemmes_caption.csv (", nrow(cap), " tokens) + ",
        "lemmes_dialogue.csv (", nrow(dia), " tokens)."
      )
    )
}

# ---------------------------------------------------------------------------
# 35b — Part relative (% de chaque source dans le total mensuel)
# ---------------------------------------------------------------------------

plot_pct_tokens <- function(df,
  titre = "Part du texte écrit vs oral dans le corpus mensuel") {

  df_long <- df |>
    select(mois, pct_caption, pct_dialogue) |>
    filter(!is.na(pct_caption)) |>
    pivot_longer(-mois, names_to = "source", values_to = "pct") |>
    mutate(source = recode(source,
      pct_caption  = "Caption (texte écrit)",
      pct_dialogue = "Dialogue (transcription Whisper)"
    ))

  pal_src <- c(
    "Caption (texte écrit)"              = unname(PAL_PHASE["1_Artisanal"]),
    "Dialogue (transcription Whisper)"   = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df_long, aes(x = mois, y = pct, fill = source)) +
    geom_phase_lines() +
    geom_area(alpha = 0.75, position = "stack") +
    scale_fill_manual(values = pal_src) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_percent(scale = 1, accuracy = 1),
      limits = c(0, 100),
      breaks = seq(0, 100, 25),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title    = titre,
      subtitle = "% de tokens par source dans le total textuel mensuel",
      x = NULL, y = "Part (%)",
      caption  = "Source : lemmes_caption.csv + lemmes_dialogue.csv."
    )
}

# ---------------------------------------------------------------------------
# 35c — Ratio legende/dialogue (ligne) avec geom_hline à 1
# ---------------------------------------------------------------------------

plot_ratio_cap_dia <- function(df,
  titre = "Ratio tokens écrits / tokens oraux par mois") {

  ggplot(df |> filter(!is.na(ratio_cap_dia)),
         aes(x = mois, y = ratio_cap_dia)) +
    geom_phase_bands() +
    geom_phase_lines() +
    geom_hline(yintercept = 1, linetype = "dotted", colour = "grey40", linewidth = 0.6) +
    annotate("text", x = min(df$mois, na.rm = TRUE), y = 1,
             label = "équilibre (ratio = 1)", hjust = 0, vjust = -0.5,
             size = 3, colour = "grey40", fontface = "italic") +
    geom_line(colour = PAL_PHASE["3_Institutionnel"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(colour = PAL_PHASE["3_Institutionnel"], size = 2.2, na.rm = TRUE) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.10))
    ) +
    labs(
      title    = titre,
      subtitle = "Ratio > 1 : legende domine | Ratio < 1 : parole domine",
      x = NULL, y = "Ratio (tokens legende / tokens dialogue)",
      caption  = paste0(
        "Source : lemmes_caption.csv + lemmes_dialogue.csv. ",
        "NA = mois sans dialogue transcrit."
      )
    )
}

# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------

save_plot(
  plot_stacked_tokens(df_mois),
  file.path(OUT, "35a_tokens_stacked.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

save_plot(
  plot_pct_tokens(df_mois),
  file.path(OUT, "35b_pct_tokens.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

save_plot(
  plot_ratio_cap_dia(df_mois),
  file.path(OUT, "35c_ratio_cap_dia.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Variante ratio + changepoint
cpts_ratio <- compute_cpts(
  df_mois$ratio_cap_dia[!is.na(df_mois$ratio_cap_dia)],
  df_mois$mois[!is.na(df_mois$ratio_cap_dia)]
)

if (length(cpts_ratio) > 0) {
  p <- add_cpt_lines(
    plot_ratio_cap_dia(df_mois, "Ratio écrit/oral — ruptures détectées"),
    cpts_ratio, color = "firebrick"
  )
  save_plot(p, file.path(OUT, "35c_ratio_cap_dia_cpt.png"),
            format = "wide_16_9", width = 10, dpi = 600)
}

# ---------------------------------------------------------------------------
# Résumé par phase
# ---------------------------------------------------------------------------

phase_labels <- c("1_Artisanal", "2_Semi-pro", "3_Institutionnel")
# Bornes de phases sourcées depuis r_source.R (source unique = config.yaml)
phase_bounds <- list(
  "1_Artisanal"      = bornes$p1,
  "2_Semi-pro"       = bornes$p2,
  "3_Institutionnel" = bornes$p3
)

df_phase_ratio <- purrr::map_dfr(phase_labels, function(ph) {
  bounds <- phase_bounds[[ph]]
  sub <- df_mois |>
    filter(mois >= bounds[1], mois <= bounds[2])
  tibble(
    phase           = ph,
    mois_n          = nrow(sub),
    tokens_cap_tot  = sum(sub$tokens_caption,  na.rm = TRUE),
    tokens_dia_tot  = sum(sub$tokens_dialogue, na.rm = TRUE),
    ratio_global    = round(
      sum(sub$tokens_caption, na.rm = TRUE) / pmax(sum(sub$tokens_dialogue, na.rm = TRUE), 1),
      3
    ),
    pct_cap_global  = round(
      sum(sub$tokens_caption, na.rm = TRUE) /
        pmax(sum(sub$tokens_caption, na.rm = TRUE) + sum(sub$tokens_dialogue, na.rm = TRUE), 1)
      * 100, 1
    )
  )
})

message("\n=== Ratio legende/dialogue par phase ===")
print(as.data.frame(df_phase_ratio))


# ---------------------------------------------------------------------------
# 35d — Ratio par phase (un panneau par phase)
# ---------------------------------------------------------------------------

PHASE_NAMES <- c(
  "1_Artisanal"      = "P1 — Artisanal (sept. 2022 – déc. 2023)",
  "2_Semi-pro"       = "P2 — Semi-pro (jan. 2024 – sept. 2024)",
  "3_Institutionnel" = "P3 — Institutionnel (oct. 2024 – sept. 2025)"
)

df_phases_all <- purrr::map_dfr(names(phase_bounds), function(ph) {
  bounds <- phase_bounds[[ph]]
  df_mois |>
    filter(mois >= bounds[1], mois <= bounds[2]) |>
    mutate(phase = ph, phase_label = PHASE_NAMES[ph])
})

plot_ratio_per_phase <- function(df, cpts_list = NULL) {
  p <- ggplot(df |> filter(!is.na(ratio_cap_dia)),
              aes(x = mois, y = ratio_cap_dia)) +
    geom_hline(yintercept = 1, linetype = "dotted",
               colour = "grey40", linewidth = 0.6) +
    geom_line(colour = PAL_PHASE["3_Institutionnel"],
              linewidth = 0.8, na.rm = TRUE) +
    geom_point(colour = PAL_PHASE["3_Institutionnel"],
               size = 2.2, na.rm = TRUE) +
    facet_wrap(~ phase_label, scales = "free_x", ncol = 1) +
    scale_x_date(date_labels = "%b %Y", date_breaks = "2 months") +
    scale_y_continuous(
      labels = label_number(accuracy = 0.1),
      expand = expansion(mult = c(0.05, 0.15))
    ) +
    labs(
      title    = "Ratio tokens écrits / tokens oraux — par phase",
      subtitle = "Ratio > 1 : legende domine | Ratio < 1 : parole domine",
      x = NULL, y = "Ratio (tokens legende / tokens dialogue)",
      caption  = "Source : lemmes_caption.csv + lemmes_dialogue.csv. NA = mois sans dialogue transcrit."
    )

  if (!is.null(cpts_list)) {
    for (ph in names(cpts_list)) {
      cpts <- cpts_list[[ph]]
      if (length(cpts) > 0) {
        cpt_df <- tibble(
          mois         = cpts,
          phase_label  = PHASE_NAMES[ph]
        )
        p <- p + geom_vline(
          data     = cpt_df,
          aes(xintercept = mois),
          colour   = "firebrick", linetype = "dashed", linewidth = 0.7
        )
      }
    }
  }
  p
}

# Calcul des changepoints par phase
cpts_per_phase <- purrr::map(names(phase_bounds), function(ph) {
  sub <- df_phases_all |>
    filter(phase == ph, !is.na(ratio_cap_dia))
  if (nrow(sub) < 4) return(as.Date(character(0)))
  compute_cpts(sub$ratio_cap_dia, sub$mois)
}) |> setNames(names(phase_bounds))

# Sans changepoint
ggsave(
  file.path(OUT, "35d_ratio_par_phase.png"),
  plot_ratio_per_phase(df_phases_all),
  width = 8, height = 11, units = "in", dpi = 600, bg = "white"
)
message("\u2714 Sauvé: 35d_ratio_par_phase.png")

# Avec changepoint
ggsave(
  file.path(OUT, "35d_ratio_par_phase_cpt.png"),
  plot_ratio_per_phase(df_phases_all, cpts_list = cpts_per_phase) +
    labs(title = "Ratio écrit/oral par phase — ruptures détectées"),
  width = 8, height = 11, units = "in", dpi = 600, bg = "white"
)
message("\u2714 Sauvé: 35d_ratio_par_phase_cpt.png")

message("=== Terminé : exports dans ", OUT, " ===")
