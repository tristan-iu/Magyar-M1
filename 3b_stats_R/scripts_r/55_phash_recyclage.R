# 55_phash_recyclage.R — Recyclage et templates visuels via perceptual hash
# Produit : 4_data_et_viz/{55a_phash_heatmap, 55b_phash_originalite_mois, 55c_phash_diversite_mois}.png
# Rscript 3b_stats_R/scripts_r/55_phash_recyclage.R

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

# Seuil Hamming pour "messages visuellement proches" (sur 64 bits).
# Convention ImageHash : ≤5 = quasi-identique, ≤10 = très proche, ≤16 = ressemblant.
# On retient 10 comme compromis entre rigueur et permissivité.
SEUIL_HAMMING <- 10L

# ---------------------------------------------------------------------------
# Préparation : filtrage et décomposition des hashes en bits
# ---------------------------------------------------------------------------

df_h <- df_clean |>
  filter(!is.na(perceptual_hash), nchar(perceptual_hash) == 16L,
         !is.na(phase), !is.na(mois)) |>
  arrange(date) |>
  mutate(
    rang_chrono = row_number(),
    phase_lbl = factor(case_when(
      phase == 1L ~ "1_Artisanal",
      phase == 2L ~ "2_Semi-pro",
      phase == 3L ~ "3_Institutionnel"
    ), levels = names(PAL_PHASE))
  )

N <- nrow(df_h)
message("=== 55_phash_recyclage.R — ", N, " messages avec hash ===")

# On décompose chaque hash hexadécimal (16 chars = 64 bits) en vecteur 0/1.
# intToBits(15)[1:4] renvoie les 4 LSB. L'ordre des bits n'affecte pas la
# distance de Hamming (on compte seulement les différences).
hex_to_bits <- function(h) {
  chars <- strsplit(h, "")[[1]]
  as.integer(unlist(lapply(chars, function(c)
    as.integer(intToBits(strtoi(c, 16L)))[1:4])))
}

BITS <- do.call(rbind, lapply(df_h$perceptual_hash, hex_to_bits))
stopifnot(ncol(BITS) == 64L, nrow(BITS) == N)

# Distance de Hamming = distance Manhattan sur vecteurs 0/1 (somme des |x_i - y_i|).
# `dist()` exploite C interne, beaucoup plus rapide qu'une boucle R.
D <- as.matrix(dist(BITS, method = "manhattan"))
storage.mode(D) <- "integer"
message("  Matrice distances : ", nrow(D), "x", ncol(D),
        " (max=", max(D), ", min hors diag=", min(D[D > 0]), ")")

# Positions des transitions de phase dans l'ordre chronologique
transitions <- which(diff(as.integer(df_h$phase)) != 0L)
phase_bounds <- c(0L, transitions, N)
phase_mids   <- (phase_bounds[-length(phase_bounds)] + phase_bounds[-1]) / 2
phase_lbls   <- paste0("P", unique(df_h$phase))

# ---------------------------------------------------------------------------
# 55a — Heatmap N×N : distance Hamming par paire, ordre chronologique
# ---------------------------------------------------------------------------

# Cap visuel à 32 bits : au-delà, les images sont "totalement différentes" et
# la nuance de couleur n'apporte rien. On garde la palette pour 0–32.
df_heat <- data.frame(
  i = rep(seq_len(N), times = N),
  j = rep(seq_len(N), each = N),
  d = as.vector(D)
)
df_heat$d_cap <- pmin(df_heat$d, 32L)

p55a <- ggplot(df_heat, aes(x = i, y = j, fill = d_cap)) +
  geom_raster() +
  scale_fill_gradient(
    low = PAL_PHASE["1_Artisanal"], high = "grey92",
    limits = c(0, 32), name = "Hamming\n(bits diff.)",
    breaks = c(0, SEUIL_HAMMING, 16, 32),
    labels = c("0 (id.)", paste0("≤", SEUIL_HAMMING), "16", "≥32")
  ) +
  geom_hline(yintercept = transitions + 0.5,
             colour = "grey25", linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = transitions + 0.5,
             colour = "grey25", linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(name = "Message (ordre chronologique)",
                     breaks = phase_mids, labels = phase_lbls,
                     expand = c(0, 0)) +
  scale_y_continuous(name = "Message (ordre chronologique)",
                     breaks = phase_mids, labels = phase_lbls,
                     expand = c(0, 0)) +
  coord_fixed() +
  labs(
    title = "Similarité visuelle entre messages (perceptual hash)",
    subtitle = "Distance de Hamming par paire. Taches bleues hors diagonale = recyclage visuel.",
    caption = paste0("Source : perceptual_hash 64-bit (pHash) sur médias. N = ", N, " messages.")
  ) +
  theme(panel.grid = element_blank(), legend.position = "right")

ggsave(file.path(OUT, "55a_phash_heatmap.png"),
       p55a, width = 12, height = 12, units = "in", dpi = 300, bg = "white")
message("✔ Sauvé: 55a_phash_heatmap.png")

# ---------------------------------------------------------------------------
# 55b — Originalité par mois : distance min au passé
# ---------------------------------------------------------------------------

# Pour chaque message i (chronologique), distance Hamming minimale à
# n'importe quel message strictement antérieur. Faible = template recyclé,
# haute = visuel inédit. Premier message exclu (pas de passé).
min_to_past <- vapply(seq_len(N), function(i) {
  if (i == 1L) return(NA_integer_)
  min(D[i, 1:(i - 1L)])
}, integer(1))

df_h$min_hamming_passe <- min_to_past

df_orig_mois <- df_h |>
  filter(!is.na(min_hamming_passe)) |>
  group_by(mois) |>
  summarise(
    med = median(min_hamming_passe),
    q1  = quantile(min_hamming_passe, 0.25),
    q3  = quantile(min_hamming_passe, 0.75),
    n   = n(),
    .groups = "drop"
  ) |>
  mutate(mois = as.Date(mois))

p55b <- ggplot(df_orig_mois, aes(x = mois)) +
  geom_phase_bands() +
  geom_phase_lines() +
  geom_hline(yintercept = SEUIL_HAMMING, colour = "firebrick",
             linetype = "dotted", linewidth = 0.5) +
  annotate("text", x = min(df_orig_mois$mois), y = SEUIL_HAMMING,
           label = paste0(" seuil recyclage (\u2264", SEUIL_HAMMING, ")"),
           hjust = 0, vjust = -0.4, size = 3,
           colour = "firebrick", fontface = "italic") +
  geom_ribbon(aes(ymin = q1, ymax = q3),
              fill = PAL_PHASE["1_Artisanal"], alpha = 0.18) +
  geom_line(aes(y = med), colour = PAL_PHASE["1_Artisanal"], linewidth = 0.8) +
  geom_point(aes(y = med), colour = PAL_PHASE["1_Artisanal"], size = 1.6) +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(name = "Distance Hamming au plus proche message antérieur (bits)",
                     breaks = seq(0, 32, 4)) +
  labs(
    title = "Originalité visuelle par mois",
    subtitle = "Médiane (\u00b1 IQR) de la distance au plus proche message antérieur. Faible = template recyclé.",
    caption = "Source : perceptual_hash. Distance < 10 bits \u2248 image quasi-identique."
  )

save_plot(p55b, file.path(OUT, "55b_phash_originalite_mois.png"),
          format = "wide_16_9", width = 10)

# ---------------------------------------------------------------------------
# 55c — Diversité visuelle intra-mois (Hamming pairwise moyen)
# ---------------------------------------------------------------------------

# Pour chaque mois ≥3 messages, moyenne des distances pairwise.
# Faible = mois "homogène" (mêmes templates), élevé = visuels variés.
calc_div_intra <- function(rangs) {
  if (length(rangs) < 2L) return(NA_real_)
  sub <- D[rangs, rangs]
  mean(sub[upper.tri(sub)])
}

div_intra <- df_h |>
  group_by(mois) |>
  summarise(
    n = n(),
    div_moy = calc_div_intra(rang_chrono),
    .groups = "drop"
  ) |>
  filter(n >= 3L) |>
  mutate(mois = as.Date(mois))

p55c <- ggplot(div_intra, aes(x = mois, y = div_moy)) +
  geom_phase_bands() +
  geom_phase_lines() +
  geom_line(colour = PAL_PHASE["2_Semi-pro"], linewidth = 0.8) +
  geom_point(aes(size = n), colour = PAL_PHASE["2_Semi-pro"]) +
  scale_size_continuous(range = c(1.5, 4), name = "n msg/mois") +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(name = "Distance Hamming moyenne intra-mois (bits)") +
  labs(
    title = "Diversit\u00e9 visuelle intra-mois",
    subtitle = "Moyenne des distances Hamming entre toutes paires de messages du mois.",
    caption = "Source : perceptual_hash. Mois \u22653 messages. Faible = templates r\u00e9p\u00e9titifs ; haute = visuels vari\u00e9s."
  )

save_plot(p55c, file.path(OUT, "55c_phash_diversite_mois.png"),
          format = "wide_16_9", width = 10)

# ---------------------------------------------------------------------------
# CSV d'appui (exploration qualitative aval)
# ---------------------------------------------------------------------------

# Pour chaque message, son plus proche "jumeau" antérieur (rang, id, distance).
twins <- vapply(seq_len(N), function(i) {
  if (i == 1L) return(NA_integer_)
  which.min(D[i, 1:(i - 1L)])
}, integer(1))

df_twins <- df_h |>
  mutate(
    jumeau_rang     = twins,
    jumeau_msg_id   = df_h$message_id[twins],
    jumeau_date     = df_h$date[twins],
    jumeau_distance = min_hamming_passe
  ) |>
  select(message_id, date, phase, mois, jumeau_msg_id, jumeau_date, jumeau_distance)

write.csv(df_twins, file.path(OUT, "55_phash_jumeaux.csv"), row.names = FALSE)
message("\u2714 Sauv\u00e9: 55_phash_jumeaux.csv (", nrow(df_twins), " lignes)")

# Synthèse mensuelle
df_synth <- df_orig_mois |>
  rename(originalite_med = med, originalite_q1 = q1, originalite_q3 = q3) |>
  left_join(div_intra |> select(mois, diversite_intra = div_moy), by = "mois")

write.csv(df_synth, file.path(OUT, "55_phash_resume_mois.csv"), row.names = FALSE)
message("\u2714 Sauv\u00e9: 55_phash_resume_mois.csv")

message("=== 55_phash_recyclage.R termin\u00e9 ===")
