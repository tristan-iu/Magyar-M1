# 38_views_mean_p1_line.R — Vues moyennes/post par mois, Phase 1, courbe + CPT
# Produit : 4_data_et_viz/38_views_mean_p1_line.png
# Rscript 3b_stats_R/scripts_r/38_views_mean_p1_line.R

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
message("=== 38_views_mean_p1_line.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# DONNÉES P1
# ---------------------------------------------------------------------------

df_p1 <- df_phase(1) %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  group_by(mois) %>%
  summarise(
    vues_totales  = sum(vues, na.rm = TRUE),
    n_posts       = n(),
    vues_moy_post = vues_totales / n_posts,
    .groups       = "drop"
  ) %>%
  complete(
    # Bornes P1 mensualisées sourcées depuis r_source.R
    mois = seq(floor_date(bornes$p1[1], "month"),
               floor_date(bornes$p1[2], "month"), by = "month"),
    fill = list(vues_totales = 0, n_posts = 0, vues_moy_post = NA_real_)
  ) %>%
  arrange(mois)

# Changepoints
cpts <- compute_cpts(df_p1$vues_moy_post, df_p1$mois)
message("Changepoints vues moy P1 : ", paste(format(cpts), collapse = ", "))

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

# Séquence de ticks : un par mois, aligné au 1er
all_months <- seq(floor_date(bornes$p1[1], "month"),
                  floor_date(bornes$p1[2], "month"), by = "month")

p <- ggplot(df_p1, aes(x = mois, y = vues_moy_post)) +
  geom_area(fill = PAL_PHASE["1_Artisanal"], alpha = 0.15, na.rm = TRUE) +
  geom_line(colour = PAL_PHASE["1_Artisanal"], linewidth = 0.9, na.rm = TRUE) +
  geom_point(aes(size = n_posts), colour = PAL_PHASE["1_Artisanal"],
             fill = "white", shape = 21, stroke = 0.8, na.rm = TRUE) +
  scale_x_date(
    breaks = all_months,
    labels = function(x) {
      ifelse(month(x) %in% c(1, 4, 7, 10),
             format(x, "%b\n%Y"),
             format(x, "%b"))
    },
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    labels = label_number(scale_cut = cut_short_scale()),
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_size_continuous(range = c(2, 5.5), name = "N posts/mois") +
  labs(
    title    = "Vues moyennes par publication (sept. 2022 – déc. 2023)",
    subtitle = "Courbe mensuelle — taille des points = nombre de publications",
    x = NULL,
    y = "Vues moyennes / publication",
    caption  = "Vues = compteur cumulatif Telegram (instantané Telegram)."
  )

p <- add_cpt_lines(p, cpts)

save_plot(p, file.path(OUT, "38_views_mean_p1_line.png"),
          format = "wide_16_9", width = 10, dpi = 600)

message("=== Terminé ===")
