# 31_correlation.R — Matrice de corrélation Spearman (corrplot)
# Produit : 4_data_et_viz/31_corrplot_spearman.png, 31_correlation_matrix.csv
# Rscript 3b_stats_R/scripts_r/31_correlation.R
# Requiert : install.packages("corrplot")

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
message("=== 31_correlation.R — ", nrow(df_clean), " messages chargés ===")

if (!requireNamespace("corrplot", quietly = TRUE))
  stop("Package 'corrplot' requis : install.packages('corrplot')")
library(corrplot)

# ---------------------------------------------------------------------------
# VARIABLES NUMÉRIQUES
# ---------------------------------------------------------------------------

df_work <- df_clean %>%
  mutate(caption_nchar = stri_length(coalesce(as.character(legende), "")))

candidate_vars <- c(
  "vues", "transferts", "reactions", "duree_sec", "caption_nchar",
  "fps", "parole_ratio", "aspect_ratio", "fichier_taille",
  "n_reaction_types", "largeur", "hauteur",
  "scene_coupes_par_min", "visages_magyar_ratio", "visages_densite"
)

var_labels <- c(
  vues = "Vues", transferts = "Forwards", reactions = "Réactions",
  duree_sec = "Durée (s)", caption_nchar = "Caption (car.)",
  fps = "FPS", parole_ratio = "Speech ratio",
  aspect_ratio = "Ratio aspect", fichier_taille = "Taille fichier",
  n_reaction_types = "Nb types react.", largeur = "Largeur px",
  hauteur = "Hauteur px", scene_coupes_par_min = "Coupes/min",
  visages_magyar_ratio = "Ratio Magyar", visages_densite = "Densité faciale"
)

available_vars <- candidate_vars[candidate_vars %in% names(df_work)]
available_vars <- available_vars[
  sapply(available_vars, function(v) sum(!is.na(df_work[[v]])) >= 30)
]

message("Variables retenues : ", paste(available_vars, collapse = ", "))

# ---------------------------------------------------------------------------
# MATRICE DE CORRÉLATION
# ---------------------------------------------------------------------------

mat_data <- df_work %>%
  select(all_of(available_vars)) %>%
  mutate(across(everything(), as.numeric))

cor_mat <- cor(mat_data, use = "pairwise.complete.obs", method = "spearman")

display_names <- ifelse(
  available_vars %in% names(var_labels),
  var_labels[available_vars],
  available_vars
)
rownames(cor_mat) <- display_names
colnames(cor_mat) <- display_names

cor_df <- as.data.frame(cor_mat)
cor_df <- cbind(variable = rownames(cor_df), cor_df)
write.csv(cor_df, file.path(OUT, "31_correlation_matrix.csv"), row.names = FALSE)
message("\u2714 Sauvé: 31_correlation_matrix.csv")

# ---------------------------------------------------------------------------
# CORRPLOT — 600 dpi
# ---------------------------------------------------------------------------

png(file.path(OUT, "31_corrplot_spearman.png"),
    width = 3600, height = 3600, res = 600)

corrplot(
  cor_mat,
  method = "color",
  type = "upper",
  order = "hclust",
  tl.col = "grey20",
  tl.cex = 0.7,
  cl.cex = 0.7,
  addCoef.col = "grey30",
  number.cex = 0.55,
  col = colorRampPalette(c(PAL_PHASE["1_Artisanal"], "white", PAL_PHASE["2_Semi-pro"]))(200),
  title = "Corrélation Spearman — variables numériques",
  mar = c(0, 0, 2, 0)
)

dev.off()
message("\u2714 Sauvé: 31_corrplot_spearman.png")

message("=== Terminé : 1 PNG + 1 CSV dans ", OUT, " ===")
