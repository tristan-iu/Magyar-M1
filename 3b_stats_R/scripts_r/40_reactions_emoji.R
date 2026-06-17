# 40_reactions_emoji.R — Distribution des réactions (emojis) par phase
# Produit : 4_data_et_viz/40a_top_emoji_global.png, 40b_emoji_phase_stacked.png,
#           40c_engagement_mois.png, 40d_emoji_diversity_boxplot.png
# Rscript 3b_stats_R/scripts_r/40_reactions_emoji.R

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
message("=== 40_reactions_emoji.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# Préparation : df_reactions est déjà dénormalisé dans r_source.R
# On assigne les phases
# ---------------------------------------------------------------------------

df_rx <- df_reactions |>
  filter(!is.na(emoji), count > 0) |>
  mutate(
    phase = case_when(
      !is.na(jour) & jour >= bornes$p1[1] & jour <= bornes$p1[2] ~ 1L,
      !is.na(jour) & jour >= bornes$p2[1] & jour <= bornes$p2[2] ~ 2L,
      !is.na(jour) & jour >= bornes$p3[1] & jour <= bornes$p3[2] ~ 3L,
      TRUE ~ NA_integer_
    ),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel",
      TRUE ~ NA_character_
    ), levels = names(PAL_PHASE))
  ) |>
  filter(!is.na(phase_lbl))

message("  Réactions dénormalisées : ", nrow(df_rx), " lignes")

# ---------------------------------------------------------------------------
# 40a — Top emojis globaux (barplot horizontal)
# ---------------------------------------------------------------------------

top_n_emoji <- 10

df_top_global <- df_rx |>
  group_by(emoji) |>
  summarise(total = sum(count, na.rm = TRUE), .groups = "drop") |>
  slice_max(total, n = top_n_emoji) |>
  mutate(emoji = fct_reorder(emoji, total))

plot_top_emoji <- function(df, titre = "Top 10 emojis de réaction") {
  ggplot(df, aes(x = total, y = emoji)) +
    geom_col(fill = PAL_PHASE["1_Artisanal"], alpha = 0.85) +
    scale_x_continuous(
      labels = label_number(scale_cut = cut_short_scale()),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = paste0("Somme des compteurs Telegram — ",
                        sum(df$total), " réactions au total"),
      x = "Total réactions", y = NULL,
      caption  = "Source : messages_clean.jsonl (reactions_detail dénormalisé)."
    )
}

save_plot(
  plot_top_emoji(df_top_global),
  file.path(OUT, "40a_top_emoji_global.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 40b — Part relative des top emojis par phase (stacked 100%)
# ---------------------------------------------------------------------------

top_emojis <- df_top_global$emoji |> as.character()

df_phase_emoji <- df_rx |>
  mutate(emoji_grp = if_else(emoji %in% top_emojis, emoji, "Autres")) |>
  group_by(phase_lbl, emoji_grp) |>
  summarise(total = sum(count, na.rm = TRUE), .groups = "drop") |>
  group_by(phase_lbl) |>
  mutate(pct = total / sum(total)) |>
  ungroup() |>
  mutate(emoji_grp = fct_reorder(emoji_grp, total, .fun = sum, .desc = TRUE))

# Garder max 5 catégories + Autres = 6 max pour PAL_CAT
top5 <- df_phase_emoji |>
  group_by(emoji_grp) |>
  summarise(s = sum(total), .groups = "drop") |>
  filter(emoji_grp != "Autres") |>
  slice_max(s, n = 5) |>
  pull(emoji_grp) |>
  as.character()

df_phase_emoji <- df_phase_emoji |>
  mutate(emoji_grp = if_else(emoji_grp %in% top5, emoji_grp, "Autres")) |>
  group_by(phase_lbl, emoji_grp) |>
  summarise(total = sum(total), .groups = "drop") |>
  group_by(phase_lbl) |>
  mutate(pct = total / sum(total)) |>
  ungroup() |>
  mutate(emoji_grp = fct_reorder(emoji_grp, total, .fun = sum, .desc = TRUE))

plot_emoji_phase <- function(df) {
  ggplot(df, aes(x = phase_lbl, y = pct, fill = emoji_grp)) +
    geom_col(position = "fill", colour = "white", linewidth = 0.3) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_fill_cat() +
    labs(
      title    = "Répartition des emojis de réaction par phase",
      subtitle = "Part relative (top 6 emojis + Autres)",
      x = NULL, y = "Proportion (%)",
      caption  = "Source : messages_clean.jsonl. reactions_detail dénormalisé."
    )
}

save_plot(
  plot_emoji_phase(df_phase_emoji),
  file.path(OUT, "40b_emoji_phase_stacked.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 40c — Réactions moyennes par post par mois (engagement)
# ---------------------------------------------------------------------------

df_engagement <- df_clean |>
  filter(!is.na(date), !is.na(mois)) |>
  mutate(mois = as.Date(mois)) |>
  group_by(mois) |>
  summarise(
    reactions_moy = mean(reactions, na.rm = TRUE),
    forwards_moy  = mean(transferts, na.rm = TRUE),
    n_posts       = n(),
    .groups = "drop"
  ) |>
  complete(mois = seq(min(mois), max(mois), by = "month"))

plot_engagement_mois <- function(df, titre = "Engagement moyen par post") {
  pal_eng <- c(
    "Réactions / post" = unname(PAL_PHASE["2_Semi-pro"]),
    "Forwards / post"  = unname(PAL_PHASE["3_Institutionnel"])
  )

  df |>
    select(mois, reactions_moy, forwards_moy) |>
    pivot_longer(-mois, names_to = "serie", values_to = "val") |>
    mutate(serie = recode(serie,
      reactions_moy = "Réactions / post",
      forwards_moy  = "Forwards / post"
    )) |>
    ggplot(aes(x = mois, y = val, colour = serie)) +
    geom_phase_lines() +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 1.5, na.rm = TRUE) +
    scale_colour_manual(values = pal_eng) +
    scale_x_mois(breaks = "3 months") +
    scale_y_continuous(
      labels = label_number(scale_cut = cut_short_scale()),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title    = titre,
      subtitle = "Moyenne mensuelle des réactions et transferts par post",
      x = NULL, y = "Moyenne / post",
      caption  = "Source : messages_clean.jsonl. Compteurs Telegram (snapshot scraping)."
    )
}

save_plot(
  plot_engagement_mois(df_engagement),
  file.path(OUT, "40c_engagement_mois.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# ---------------------------------------------------------------------------
# 40d — Nombre de types d'emojis distincts par phase (diversité réactive)
# ---------------------------------------------------------------------------

df_diversity <- df_rx |>
  group_by(phase_lbl, message_id) |>
  summarise(n_types = n_distinct(emoji), .groups = "drop")

med_div <- df_diversity |>
  group_by(phase_lbl) |>
  summarise(med = median(n_types), n = n(), .groups = "drop")

plot_emoji_diversity <- function() {
  ggplot(df_diversity, aes(x = phase_lbl, y = n_types, fill = phase_lbl)) +
    geom_boxplot(width = 0.55, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(data = med_div,
              aes(x = phase_lbl, y = med,
                  label = sprintf("méd. = %d", med)),
              vjust = -0.8, size = 3.8, fontface = "bold") +
    scale_fill_phase(short = TRUE) +
    scale_x_discrete(labels = LBL_PHASE_SHORT) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.12))) +
    labs(
      title    = "Diversité des réactions par phase",
      subtitle = "Nombre d'emojis distincts par post",
      x = NULL, y = "Nb types d'emojis",
      caption  = paste0(
        "Source : messages_clean.jsonl. ",
        paste(sprintf("%s : n=%d posts", LBL_PHASE_SHORT[as.character(med_div$phase_lbl)],
              med_div$n), collapse = " | "), "."
      )
    ) +
    guides(fill = "none")
}

save_plot(
  plot_emoji_diversity(),
  file.path(OUT, "40d_emoji_diversity_boxplot.png"),
  format = "wide_16_9", width = 10, dpi = 600
)

# Résumé
message("\n=== Top emojis globaux ===")
print(as.data.frame(df_top_global))
message("\n=== Diversité réactions par phase ===")
print(as.data.frame(med_div))
message("=== Terminé : exports dans ", OUT, " ===")
