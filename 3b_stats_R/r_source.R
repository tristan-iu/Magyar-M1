library(jsonlite)
library(dplyr)
library(purrr)
library(lubridate)
library(tidyr)
library(ggplot2)
library(scales)
library(forcats)
library(stringr)

# Chemin corpus local — seul réglage machine-spécifique de tout le module
# (corpus non inclus dans le dépôt, disponible sur demande)
chemin_jsonl <- "/media/elwin/NVME-div/M1_corpus/processed/messages_clean.jsonl"

df_brut <- stream_in(file(chemin_jsonl), flatten = TRUE)

# Nettoyage des données
ensure_col <- function(df, col, default) {
  if (!col %in% names(df)) df[[col]] <- default
  df
}

parse_date_safe <- function(x, tz = "UTC") {
  if (inherits(x, "POSIXt")) return(as.POSIXct(x, tz = tz))

  x <- as.character(x)
  x <- trimws(x)
  x[x == ""] <- NA_character_

  # 1) Normaliser "YYYY-MM-DDTHHMMSS" -> "YYYY-MM-DD HH:MM:SS"
  x2 <- gsub("T", " ", x, fixed = TRUE)
  x2 <- sub("(\\d{4}-\\d{2}-\\d{2}) (\\d{2})(\\d{2})(\\d{2})$",
            "\\1 \\2:\\3:\\4", x2, perl = TRUE)

  # 2) Normaliser "YYYY-MM-DD HHMM" -> "YYYY-MM-DD HH:MM:00"
  x2 <- sub("(\\d{4}-\\d{2}-\\d{2}) (\\d{2})(\\d{2})$",
            "\\1 \\2:\\3:00", x2, perl = TRUE)

  out <- suppressWarnings(lubridate::ymd_hms(x2, tz = tz, quiet = TRUE))
  miss <- is.na(out) & !is.na(x2)
  if (any(miss)) out[miss] <- suppressWarnings(lubridate::ymd_hm(x2[miss], tz = tz, quiet = TRUE))
  miss <- is.na(out) & !is.na(x2)
  if (any(miss)) out[miss] <- suppressWarnings(lubridate::ymd(x2[miss], tz = tz, quiet = TRUE))

  out
}

nettoyer_messages <- function(df_brut) {

  # Colonnes manquantes (corpus partiellement enrichi) créées en NA
  df_brut <- df_brut |>
    ensure_col("message_id",       NA_integer_)   |>
    ensure_col("canal",             NA_character_) |>
    ensure_col("date",              NA_character_) |>
    ensure_col("legende",           NA_character_) |>
    ensure_col("vues",              NA_real_)      |>
    ensure_col("transferts",        NA_real_)      |>
    ensure_col("reactions",         NA_real_)      |>
    ensure_col("media_type",        NA_character_) |>
    ensure_col("media_chemin",      NA_character_) |>
    ensure_col("duree",             NA_real_)      |>
    ensure_col("fps",               NA_real_)      |>
    ensure_col("video_bitrate",     NA_real_)      |>
    ensure_col("fichier_taille",    NA_real_)      |>
    ensure_col("parole_ratio",      NA_real_)      |>
    ensure_col("reactions_detail",  vector("list", nrow(df_brut))) |>
    ensure_col("largeur",           NA_integer_)   |>
    ensure_col("hauteur",           NA_integer_)   |>
    ensure_col("orientation",       NA_character_) |>
    ensure_col("album_id",          NA_character_) |>
    ensure_col("album_rang",        NA_integer_)

  df_brut |>
    mutate(
      # IDs / types
      message_id = suppressWarnings(as.integer(message_id)),
      canal      = as.character(canal),

      # album_id peut être gigantesque -> texte (sinon overflow int64)
      album_id   = as.character(album_id),
      album_rang = suppressWarnings(as.integer(album_rang)),

      # date : parsing robuste + colonnes temporelles
      date_raw = date,
      date     = parse_date_safe(date, tz = "UTC"),
      jour     = as.Date(date),
      semaine  = floor_date(date, "week"),
      mois     = floor_date(date, "month"),

      # légende Telegram
      legende = na_if(trimws(as.character(legende)), ""),

      # métriques d'engagement : on capture le flag "manquant" AVANT imputation
      vues_is_missing       = is.na(suppressWarnings(as.numeric(vues))),
      transferts_is_missing = is.na(suppressWarnings(as.numeric(transferts))),
      reactions_is_missing  = is.na(suppressWarnings(as.numeric(reactions))),

      vues       = replace_na(suppressWarnings(as.numeric(vues)),       0),
      transferts = replace_na(suppressWarnings(as.numeric(transferts)), 0),
      reactions  = replace_na(suppressWarnings(as.numeric(reactions)),  0),

      # media_type : normalisation
      media_type   = as.character(media_type),
      media_type   = if_else(is.na(media_type) | media_type == "",
                             "texte_uniquement", media_type),
      media_chemin = as.character(media_chemin),
      has_media    = (media_type != "texte_uniquement") | !is.na(media_chemin),

      # durée vidéo (ffprobe)
      duree     = suppressWarnings(as.numeric(duree)),
      duree_sec = coalesce(duree, 0),

      # autres métriques techniques
      fps             = suppressWarnings(as.numeric(fps)),
      video_bitrate   = suppressWarnings(as.numeric(video_bitrate)),
      fichier_taille  = suppressWarnings(as.numeric(fichier_taille)),

      # parole / dialogue — fusion pour éviter double assignation
      parole_ratio = replace_na(suppressWarnings(as.numeric(parole_ratio)), 0)
    ) |>
    # colonnes listes et dimensions image
    mutate(
      reactions_detail = map(reactions_detail,
                             function(x) if (is.null(x)) list() else x),
      n_reaction_types = map_int(reactions_detail, length),

      largeur      = suppressWarnings(as.integer(largeur)),
      hauteur      = suppressWarnings(as.integer(hauteur)),
      aspect_ratio = largeur / hauteur,

      orientation = case_when(
        !is.na(orientation) & orientation != "" ~ as.character(orientation),
        !is.na(largeur) & !is.na(hauteur) & largeur == hauteur ~ "square",
        !is.na(largeur) & !is.na(hauteur) & largeur >  hauteur ~ "horizontal",
        !is.na(largeur) & !is.na(hauteur) & largeur <  hauteur ~ "vertical",
        TRUE ~ NA_character_
      )
    ) |>
    # dédoublonnage : on garde la ligne "la plus informative"
    arrange(canal, message_id, desc(!is.na(date)),
            desc(vues), desc(transferts), desc(reactions), desc(has_media)) |>
    distinct(canal, message_id, .keep_all = TRUE)
}

df_clean <- nettoyer_messages(df_brut)

# Phases
# Borne P1[1] = 2022-09-01 : 7 messages (id 1-7) postés ce jour-là, avant la
# date « officielle » du 2022-09-02 (premier post structuré). On les inclut dans P1.
bornes <- list(
  p1 = as.Date(c("2022-09-01", "2023-12-31")),
  p2 = as.Date(c("2024-01-01", "2024-09-30")),
  p3 = as.Date(c("2024-10-01", "2025-09-30"))
)

assigner_phase <- function(df) {
  df %>%
    mutate(
      jour = as.Date(date),
      phase = case_when(
        !is.na(jour) & jour >= bornes$p1[1] & jour <= bornes$p1[2] ~ 1L,
        !is.na(jour) & jour >= bornes$p2[1] & jour <= bornes$p2[2] ~ 2L,
        !is.na(jour) & jour >= bornes$p3[1] & jour <= bornes$p3[2] ~ 3L,
        TRUE ~ NA_integer_
      )
    )
}

df_clean <- assigner_phase(df_clean)

# Sous-corpus P3 du canal perso. Utilisé par 36_overlay_posts_duree_avg.R.
# Sert aussi de base aux analyses cross-canal, qui chargent le corpus officiel
# @magyarbirds414 et définissent df_cross dans 414_comparaison/r_source_birds.R
# (sourcé uniquement par ces scripts, après ce fichier).
df_p3 <- df_clean %>% dplyr::filter(phase == 3L)

df_phase <- function(x = c(1L, 2L, 3L), .data = df_clean) {
  x <- as.integer(x)
  dplyr::filter(.data, phase %in% x)
}

df_p1  <- df_phase(1)
df_p12 <- df_phase(c(1, 2))

# Table réactions normalisées
df_reactions <- df_clean %>%
  select(canal, message_id, date, jour, semaine, mois, media_type, vues, transferts, reactions, reactions_detail) %>%
  unnest_longer(reactions_detail, values_to = "reaction") %>%
  unnest_wider(reaction) %>%
  mutate(
    emoji = as.character(emoji),
    count = suppressWarnings(as.numeric(count)),
    count = replace_na(count, 0)
  )

# THEME PRINCIPAL
theme_madyar <- function(base_size = 14, base_family = "sans",
                          x_text_angle = 0, x_text_hjust = NULL) {

  if (is.null(x_text_hjust)) {
    x_text_hjust <- ifelse(x_text_angle == 0, 0.5, 1)
  }

  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey85", linewidth = 0.3),

      plot.title = element_text(
        face = "bold", size = base_size + 4, hjust = 0.5,
        margin = margin(b = 8)
      ),
      plot.subtitle = element_text(
        color = "grey30", size = base_size, hjust = 0.5,
        margin = margin(b = 12)
      ),
      plot.caption = element_text(
        color = "grey35", size = base_size - 3, hjust = 0,
        margin = margin(t = 10)
      ),
      plot.caption.position = "plot",
      plot.title.position = "plot",

      axis.title = element_text(face = "bold", margin = margin(t = 8, r = 8)),
      axis.text = element_text(color = "grey10"),
      axis.text.x = element_text(angle = x_text_angle, hjust = x_text_hjust),

      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 2),
      legend.margin = margin(t = 4),
      legend.box.spacing = unit(6, "pt"),

      strip.text = element_text(face = "bold", size = base_size - 1),
      strip.background = element_rect(fill = "grey95", color = NA),

      plot.margin = margin(12, 12, 12, 12)
    )
}

theme_set(theme_madyar())


# ---------------------------------------------------------------------------
# PALETTES
# ---------------------------------------------------------------------------

# Couleurs par phase : bleu → orange → gris

PAL_PHASE <- c(
  "1_Artisanal"      = "#2166AC",
  "2_Semi-pro"       = "#D6804B",
  "3_Institutionnel" = "#4D4D4D"
)

LBL_PHASE <- c(
  "1_Artisanal"      = "Artisanal (sept. 2022 – déc. 2023)",
  "2_Semi-pro"       = "Semi-pro (janv. – sept. 2024)",
  "3_Institutionnel" = "Institutionnel (oct. 2024 – juin 2025)"
)

# Version courte pour légendes compactes
LBL_PHASE_SHORT <- c(
  "1_Artisanal"      = "P1",
  "2_Semi-pro"       = "P2",
  "3_Institutionnel" = "P3"
)

# Palette catégorielle — ColorBrewer Dark2 (max 6, sinon regrouper en "Autres")
PAL_CAT <- c(
  "#1B9E77",
  "#D95F02",
  "#7570B3",
  "#E7298A",
  "#66A61E",
  "#525252"
)

# Séquentielle (heatmaps, densités) — ColorBrewer Blues
PAL_SEQ_LOW  <- "#F7FBFF"
PAL_SEQ_HIGH <- "#08519C"

# ---------------------------------------------------------------------------
# SCALES GGPLOT2
# ---------------------------------------------------------------------------

# Fill/colour par phase
scale_fill_phase <- function(short = FALSE, ...) {
  lbls <- if (short) LBL_PHASE_SHORT else LBL_PHASE
  scale_fill_manual(values = PAL_PHASE, labels = lbls, ...)
}

scale_colour_phase <- function(short = FALSE, ...) {
  lbls <- if (short) LBL_PHASE_SHORT else LBL_PHASE
  scale_colour_manual(values = PAL_PHASE, labels = lbls, ...)
}

# Fill/colour catégorielle (max 6)
scale_fill_cat <- function(...) {
  scale_fill_manual(values = PAL_CAT, ...)
}

scale_colour_cat <- function(...) {
  scale_colour_manual(values = PAL_CAT, ...)
}

# Axe X date — labels horizontaux, pas temporel constant
scale_x_mois <- function(breaks = "3 months", fmt = "%b\n%Y",
                         angle = 0, ...) {
  scale_x_date(
    date_breaks = breaks,
    date_labels = fmt,
    expand = expansion(mult = c(0.02, 0.02)),
    ...
  )
}


# ---------------------------------------------------------------------------
# LIGNES DE PHASE
# ---------------------------------------------------------------------------

PHASE_DATES <- as.Date(c("2024-01-01", "2024-10-01"))

# Ajoute des vlines pointillées + labels en haut du graphique.
# Usage : p <- p + geom_phase_lines()
geom_phase_lines <- function(dates = PHASE_DATES,
                             labels = c("P1\u2192P2", "P2\u2192P3"),
                             colour = "#888888",
                             linetype = "dashed",
                             linewidth = 0.5,
                             label_size = 3,
                             label_y = Inf,
                             label_vjust = 1.8) {
  list(
    geom_vline(
      xintercept = as.numeric(dates),
      colour = colour, linetype = linetype, linewidth = linewidth
    ),
    annotate(
      "text",
      x = dates, y = label_y,
      label = paste0("\u2190 ", labels),
      hjust = 0, vjust = label_vjust,
      size = label_size, colour = colour, fontface = "italic"
    )
  )
}

add_phase_lines <- function(p, ...) {
  p + geom_phase_lines(...)
}


# ---------------------------------------------------------------------------
# ANNOTATIONS DE FOND PAR PHASE (optionnel)
# ---------------------------------------------------------------------------
# Bandes colorées légères derrière les données.
# Utile sur les graphiques denses (heatmap, stacked bars).

geom_phase_bands <- function(bornes_list = bornes,
                             alpha = 0.06,
                             ymin = -Inf, ymax = Inf) {
  list(
    annotate("rect",
             xmin = bornes_list$p1[1], xmax = bornes_list$p1[2],
             ymin = ymin, ymax = ymax,
             fill = PAL_PHASE["1_Artisanal"], alpha = alpha),
    annotate("rect",
             xmin = bornes_list$p2[1], xmax = bornes_list$p2[2],
             ymin = ymin, ymax = ymax,
             fill = PAL_PHASE["2_Semi-pro"], alpha = alpha),
    annotate("rect",
             xmin = bornes_list$p3[1], xmax = bornes_list$p3[2],
             ymin = ymin, ymax = ymax,
             fill = PAL_PHASE["3_Institutionnel"], alpha = alpha)
  )
}


# ---------------------------------------------------------------------------
# SAVE : support SVG + PNG
# ---------------------------------------------------------------------------

save_plot <- function(plot_obj, filename,
                      format = c("wide_16_9", "square"),
                      width = 8, units = "in", dpi = 600,
                      bg = "white", device = NULL) {

  format <- match.arg(format)
  height <- if (format == "wide_16_9") width * 9/16 else width

  # Détection auto du device par extension
  if (is.null(device)) {
    ext <- tolower(tools::file_ext(filename))
    if (ext == "svg") device <- "svg"
  }

  ggsave(
    filename, plot = plot_obj,
    width = width, height = height, units = units,
    dpi = dpi, bg = bg, device = device, limitsize = FALSE
  )

  message("\u2714 Sauvé: ", filename,
          " (", format, ", ", width, "x", round(height, 2), " ", units,
          ", ", dpi, " dpi)")
  invisible(filename)
}


# ---------------------------------------------------------------------------
# VARIANTES DU THÈME
# ---------------------------------------------------------------------------

# Variante pour graphiques avec axe X catégoriel (barplots par mois-texte).
theme_madyar_bar <- function(...) {
  theme_madyar(...) +
    theme(
      axis.line.x = element_line(colour = "grey50", linewidth = 0.3),
      panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.25)
    )
}

# Variante pour facettes (heatmap, small multiples).
theme_madyar_facet <- function(...) {
  theme_madyar(...) +
    theme(
      strip.background = element_rect(fill = "white", colour = "grey80",
                                      linewidth = 0.3),
      panel.spacing = unit(8, "pt")
    )
}


# ---------------------------------------------------------------------------
# DIAGNOSTICS SÉMIOLOGIQUES
# ---------------------------------------------------------------------------

# Avertit si une variable dépasse max_n catégories (lisibilité).
check_n_categories <- function(x, max_n = 8, var_name = "variable") {
  n <- length(unique(na.omit(x)))
  if (n > max_n) {
    warning(
      sprintf("[Bertin] %s a %d catégories (seuil = %d). Regrouper en 'Autres'.",
              var_name, n, max_n),
      call. = FALSE
    )
  }
  invisible(n)
}


# ---------------------------------------------------------------------------
# HELPER : indexation min-max 0-100
# ---------------------------------------------------------------------------

to_index_100 <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) return(rep(NA_real_, length(x)))
  (x - rng[1]) / (rng[2] - rng[1]) * 100
}

# ---------------------------------------------------------------------------
# HELPER : normalise le dernier mois incomplet par extrapolation linéaire
# (nb_observés / jours_couverts × jours_dans_le_mois)
# Retourne list(df = df_normalisé, extrapol_mois = Date | NULL)
# ---------------------------------------------------------------------------

.normalize_last_month_posts <- function(df_mois, val_col, last_obs_date) {
  if (is.null(last_obs_date)) return(list(df = df_mois, extrapol_mois = NULL))
  last_month   <- as.Date(lubridate::floor_date(last_obs_date, "month"))
  if (!last_month %in% df_mois$mois) return(list(df = df_mois, extrapol_mois = NULL))
  days_covered <- as.integer(last_obs_date - last_month) + 1L
  days_total   <- as.integer(lubridate::days_in_month(last_obs_date))
  df_mois[[val_col]][df_mois$mois == last_month] <-
    df_mois[[val_col]][df_mois$mois == last_month] * days_total / days_covered
  list(df = df_mois, extrapol_mois = last_month)
}

# ---------------------------------------------------------------------------
# HELPERS CHANGEPOINT — partagés entre 26_changepoint et variantes _cpt
# Requiert : install.packages("changepoint")
# ---------------------------------------------------------------------------

# Retourne un vecteur Date des ruptures (cpt.meanvar, PELT, MBIC).
# Pénalité MBIC : conservatrice, vise 2-3 ruptures max/série.
compute_cpts <- function(vals, mois, method = "PELT", penalty = "MBIC") {
  if (!requireNamespace("changepoint", quietly = TRUE))
    stop("Package 'changepoint' requis : install.packages('changepoint')")
  tryCatch({
    clean <- !is.na(vals)
    if (sum(clean) < 5) return(as.Date(character(0)))
    res <- changepoint::cpt.meanvar(vals[clean], method = method, penalty = penalty)
    idx <- changepoint::cpts(res)
    if (length(idx) == 0) return(as.Date(character(0)))
    as.Date(mois[clean][idx])
  }, error = function(e) as.Date(character(0)))
}

# Superpose des geom_vline de rupture sur un ggplot à axe Date
add_cpt_lines <- function(p, dates,
                          color = "firebrick", linetype = "dashed", linewidth = 0.7) {
  if (length(dates) == 0) return(p)
  p + geom_vline(xintercept = as.numeric(as.Date(dates)),
                 color = color, linetype = linetype, linewidth = linewidth)
}

# ---------------------------------------------------------------------------
# CONSTANTES CORPUS
# ---------------------------------------------------------------------------

LAST_OBS <- as.Date("2025-09-30")   # dernière date du scraping (fin corpus)

# Milestone intra-phase (exploratoire — pas une borne canonique ; le canonique
# reste `bornes`). DATE_RUPTURE_FPV_P1 : bascule au sein de P1 pour l'analyse
# kamikaze (scripts_r/analyse_kamikaze_P1.R).
DATE_RUPTURE_FPV_P1 <- as.Date("2023-03-01")

# Séquence mensuelle complète couvrant un dataframe
mois_seq_df <- function(data) {
  m <- as.Date(floor_date(data$date[!is.na(data$date)], "month"))
  seq(min(m), max(m), by = "month")
}

