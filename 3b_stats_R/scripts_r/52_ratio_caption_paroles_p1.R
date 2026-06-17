# 52_ratio_caption_paroles_p1.R — Ratio texte écrit / parole : zoom P1 — JSONL vs lexico
# Produit : 4_data_et_viz/52a_stacked_nchar_p1.png, 52b/52c_ratio_*_p1.png, 52d_overlay_ratios_p1.png
# Rscript 3b_stats_R/scripts_r/52_ratio_caption_paroles_p1.R

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

LEX <- normalizePath(file.path(BASE, "..", "4_data_et_viz", "lexico"))
message("=== 52_ratio_caption_paroles_p1.R — ", nrow(df_clean), " messages chargés ===")

# Bornes P1 sourcées depuis r_source.R (source unique = config.yaml)
P1_DEBUT <- bornes$p1[1]
P1_FIN   <- bornes$p1[2]

# Séquence mensuelle complète P1 (pour complete())
mois_p1 <- seq(
  as.Date(format(P1_DEBUT, "%Y-%m-01")),
  as.Date(format(P1_FIN,   "%Y-%m-01")),
  by = "month"
)

# ── Section A — JSONL direct (nchar) ──────────────────────────────────────────

# On retient les messages P1 ayant caption ET dialogue non vides
# (exclut les text-only et les vidéos silencieuses).
df_p1 <- df_phase(1) |>
  filter(
    !is.na(legende),  nchar(trimws(legende))  > 0,
    !is.na(dialogue), nchar(trimws(dialogue)) > 0,
    !is.na(mois)
  ) |>
  mutate(
    mois          = as.Date(mois),
    ncar_legende  = nchar(legende),
    ncar_dialogue = nchar(dialogue)
  )

message("  P1 avec caption + dialogue : ", nrow(df_p1), " messages")

# On agrège par mois : somme des caractères par source
df_mois_jl <- df_p1 |>
  group_by(mois) |>
  summarise(
    ncar_cap = sum(ncar_legende,  na.rm = TRUE),
    ncar_dia = sum(ncar_dialogue, na.rm = TRUE),
    n_msg    = n(),
    .groups  = "drop"
  ) |>
  complete(mois = mois_p1, fill = list(ncar_cap = 0L, ncar_dia = 0L, n_msg = 0L)) |>
  mutate(
    ratio_jl = if_else(ncar_dia > 0, ncar_cap / ncar_dia, NA_real_)
  )

# ── 52a — Stacked area nchar caption vs dialogue (P1 mensuel) ─────────────────

plot_52a <- function(df) {
  df_long <- df |>
    select(mois, ncar_cap, ncar_dia) |>
    pivot_longer(-mois, names_to = "source", values_to = "ncar") |>
    mutate(source = recode(source,
      ncar_cap = "Caption (texte écrit)",
      ncar_dia = "Dialogue (transcription Whisper)"
    ))

  pal_src <- c(
    "Caption (texte écrit)"            = unname(PAL_PHASE["1_Artisanal"]),
    "Dialogue (transcription Whisper)" = unname(PAL_PHASE["2_Semi-pro"])
  )

  ggplot(df_long, aes(x = mois, y = ncar, fill = source)) +
    geom_area(alpha = 0.75, position = "stack") +
    scale_fill_manual(values = pal_src, name = NULL) +
    scale_x_date(date_labels = "%b %Y", date_breaks = "2 months",
                 expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(
      labels = label_number(big.mark = "\u202F"),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(
      title    = "Volume mensuel de caractères — texte écrit vs parole (P1)",
      subtitle = "nchar(legende) + nchar(dialogue), messages avec caption et dialogue non vides",
      x = NULL, y = "Nombre de caractères",
      caption  = paste0(
        "Source : messages_clean.jsonl — ", sum(df$n_msg, na.rm = TRUE),
        " messages P1 avec caption+dialogue (sept. 2022 – déc. 2023)."
      )
    )
}

save_plot(
  plot_52a(df_mois_jl),
  file.path(OUT, "52a_stacked_nchar_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ── 52b — Ratio nchar caption / dialogue par mois (JSONL) ─────────────────────

plot_ratio_jl <- function(df,
    titre = "Ratio texte écrit / parole — P1 (caractères bruts, JSONL)") {
  ggplot(df |> filter(!is.na(ratio_jl)), aes(x = mois, y = ratio_jl)) +
    geom_hline(yintercept = 1, linetype = "dotted", colour = "grey40", linewidth = 0.6) +
    annotate("text", x = min(df$mois, na.rm = TRUE), y = 1,
             label = "équilibre (ratio = 1)", hjust = 0, vjust = -0.5,
             size = 3, colour = "grey40", fontface = "italic") +
    geom_smooth(method = "loess", span = 0.6, se = FALSE, colour = "grey70",
                linewidth = 0.6, linetype = "dashed", na.rm = TRUE) +
    geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(colour = PAL_PHASE["1_Artisanal"], size = 2.2, na.rm = TRUE) +
    scale_x_date(date_labels = "%b %Y", date_breaks = "2 months",
                 expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(
      labels = label_number(accuracy = 0.01),
      expand = expansion(mult = c(0.05, 0.12))
    ) +
    labs(
      title    = titre,
      subtitle = "Ratio > 1 : caption domine | Ratio < 1 : parole domine",
      x = NULL, y = "Ratio (nchar legende / nchar dialogue)",
      caption  = "Source : messages_clean.jsonl — nchar(legende) / nchar(dialogue). NA = mois sans données."
    )
}

save_plot(
  plot_ratio_jl(df_mois_jl),
  file.path(OUT, "52b_ratio_nchar_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ── Section B — CSV lexico (tokens lemmatisés, P1 seulement) ──────────────────

# On charge les CSV lexico produits par 3a_lexicometrie/lexicometrie.py.
# Chaque ligne = un token d'un message. On filtre aux dates P1 puis on compte
# par mois — même logique que 35_ratio_texte_oral.R mais restreint à P1.
parse_lex_date <- function(x) {
  suppressWarnings(as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC"))
}

cap_raw <- read.csv(file.path(LEX, "lemmes_caption.csv"),
                    stringsAsFactors = FALSE, encoding = "UTF-8")
dia_raw <- read.csv(file.path(LEX, "lemmes_dialogue.csv"),
                    stringsAsFactors = FALSE, encoding = "UTF-8")

tokens_cap_p1 <- cap_raw |>
  mutate(
    date = parse_lex_date(date),
    mois = as.Date(floor_date(date, "month"))
  ) |>
  filter(!is.na(mois), mois >= as.Date(format(P1_DEBUT, "%Y-%m-01")),
                        mois <= as.Date(format(P1_FIN,   "%Y-%m-01"))) |>
  count(mois, name = "tokens_cap")

tokens_dia_p1 <- dia_raw |>
  mutate(
    date = parse_lex_date(date),
    mois = as.Date(floor_date(date, "month"))
  ) |>
  filter(!is.na(mois), mois >= as.Date(format(P1_DEBUT, "%Y-%m-01")),
                        mois <= as.Date(format(P1_FIN,   "%Y-%m-01"))) |>
  count(mois, name = "tokens_dia")

message("  Tokens caption P1  : ", sum(tokens_cap_p1$tokens_cap))
message("  Tokens dialogue P1 : ", sum(tokens_dia_p1$tokens_dia))

df_mois_lx <- tibble(mois = mois_p1) |>
  left_join(tokens_cap_p1, by = "mois") |>
  left_join(tokens_dia_p1, by = "mois") |>
  mutate(
    tokens_cap = replace_na(tokens_cap, 0L),
    tokens_dia = replace_na(tokens_dia, 0L),
    ratio_lx   = if_else(tokens_dia > 0, tokens_cap / tokens_dia, NA_real_)
  )

# ── 52c — Ratio tokens caption / dialogue par mois (lexico) ───────────────────

plot_ratio_lx <- function(df,
    titre = "Ratio texte écrit / parole — P1 (tokens lemmatisés, spaCy)") {
  ggplot(df |> filter(!is.na(ratio_lx)), aes(x = mois, y = ratio_lx)) +
    geom_hline(yintercept = 1, linetype = "dotted", colour = "grey40", linewidth = 0.6) +
    annotate("text", x = min(df$mois, na.rm = TRUE), y = 1,
             label = "équilibre (ratio = 1)", hjust = 0, vjust = -0.5,
             size = 3, colour = "grey40", fontface = "italic") +
    geom_smooth(method = "loess", span = 0.6, se = FALSE, colour = "grey70",
                linewidth = 0.6, linetype = "dashed", na.rm = TRUE) +
    geom_line(colour = PAL_PHASE["3_Institutionnel"], linewidth = 0.8, na.rm = TRUE) +
    geom_point(colour = PAL_PHASE["3_Institutionnel"], size = 2.2, na.rm = TRUE) +
    scale_x_date(date_labels = "%b %Y", date_breaks = "2 months",
                 expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(
      labels = label_number(accuracy = 0.01),
      expand = expansion(mult = c(0.05, 0.12))
    ) +
    labs(
      title    = titre,
      subtitle = "Ratio > 1 : caption domine | Ratio < 1 : parole domine",
      x = NULL, y = "Ratio (tokens legende / tokens dialogue)",
      caption  = paste0(
        "Source : lemmes_caption.csv + lemmes_dialogue.csv — ",
        sum(tokens_cap_p1$tokens_cap), " tokens caption / ",
        sum(tokens_dia_p1$tokens_dia), " tokens dialogue (P1)."
      )
    )
}

save_plot(
  plot_ratio_lx(df_mois_lx),
  file.path(OUT, "52c_ratio_tokens_p1.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ── Section D — Overlay indexé JSONL vs lexico ────────────────────────────────

# On normalise les deux ratios sur 0–100 (min-max) pour les rendre comparables
# sur un même axe.
df_overlay <- df_mois_jl |>
  select(mois, ratio_jl) |>
  full_join(df_mois_lx |> select(mois, ratio_lx), by = "mois") |>
  filter(!is.na(ratio_jl) | !is.na(ratio_lx)) |>
  mutate(
    idx_jl = to_index_100(ratio_jl),
    idx_lx = to_index_100(ratio_lx)
  )

if (sum(!is.na(df_overlay$idx_jl)) >= 3 && sum(!is.na(df_overlay$idx_lx)) >= 3) {
  pal_ov <- c(
    "Caractères bruts (JSONL)"  = unname(PAL_PHASE["1_Artisanal"]),
    "Tokens lemmatisés (spaCy)" = unname(PAL_PHASE["3_Institutionnel"])
  )

  p_ov <- df_overlay |>
    select(mois, idx_jl, idx_lx) |>
    pivot_longer(-mois, names_to = "methode", values_to = "idx") |>
    mutate(methode = recode(methode,
      idx_jl = "Caractères bruts (JSONL)",
      idx_lx = "Tokens lemmatisés (spaCy)"
    )) |>
    ggplot(aes(x = mois, y = idx, colour = methode)) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    scale_colour_manual(values = pal_ov, name = NULL) +
    scale_x_date(date_labels = "%b %Y", date_breaks = "2 months",
                 expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
    labs(
      title    = "Ratio écrit / oral en P1 — convergence JSONL vs lexico",
      subtitle = "Indices 0–100 (min-max mensuel) — convergence = robustesse du signal",
      x = NULL, y = "Indice (0 = min, 100 = max)",
      caption  = "Source : messages_clean.jsonl (nchar) + lemmes_caption/dialogue.csv (tokens spaCy)."
    )

  save_plot(
    p_ov,
    file.path(OUT, "52d_overlay_ratios_p1.png"),
    format = "wide_16_9", width = 10, dpi = 600
  )
}

# ── Résumé console ────────────────────────────────────────────────────────────

df_resume <- df_mois_jl |>
  select(mois, ncar_cap, ncar_dia, ratio_jl) |>
  left_join(df_mois_lx |> select(mois, tokens_cap, tokens_dia, ratio_lx), by = "mois")

message("\n=== Ratio caption/paroles P1 — par mois ===")
print(as.data.frame(df_resume))
message("=== Terminé : 4 exports dans ", OUT, " ===")
