# 34_lda_topics.R — LDA topic modelling sur le corpus lexical
# Deux granularités : par message (gamma temporel fin) + par mois (robustesse)
# Stack : text2vec (pas de dépendance libxml2/tm)
# Produit : 4_data_et_viz/34a-34e (beta, gamma temporel, gamma par phase) + 4 CSV
# Rscript 3b_stats_R/scripts_r/34_lda_topics.R
# Requiert : install.packages("text2vec")
# NOTE : la version retenue pour le mémoire est le LDA gensim (3a_lexicometrie),
#        qui fournit les scores de cohérence C_v ; ce script reste la variante R.

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

if (!requireNamespace("text2vec", quietly = TRUE))
  stop("install.packages('text2vec')")

library(text2vec)
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------

K          <- 6L
SEED       <- 42L
N_ITER     <- 1500L   # itérations variational Bayes (text2vec)
MIN_TOKENS <- 5L      # min tokens par document
MIN_TF     <- 3L      # min occurrences globales d'un terme
POS_OK     <- c("NOUN", "VERB", "ADJ", "PRON", "ADV")

TOPIC_COLORS <- colorRampPalette(c(
  PAL_PHASE["1_Artisanal"], PAL_PHASE["2_Semi-pro"],
  PAL_PHASE["3_Institutionnel"], "#6B4C9A", "#2E8B57", "#C0392B"
))(K)

LEMMES_CSV <- file.path(dirname(BASE), "4_data_et_viz", "lexico", "lemmes_combined.csv")
message("=== 34_lda_topics.R — chargement ", LEMMES_CSV, " ===")

# ---------------------------------------------------------------------------
# 1. CHARGEMENT & FILTRAGE
# ---------------------------------------------------------------------------

df_raw <- read_csv(LEMMES_CSV, show_col_types = FALSE) %>%
  filter(pos %in% POS_OK, !is.na(lemma), lemma != "") %>%
  mutate(
    date  = as.POSIXct(date, tz = "UTC"),
    jour  = as.Date(date),
    mois  = as.Date(floor_date(date, "month")),
    # Bornes de phases sourcées depuis r_source.R (source unique = config.yaml)
    phase = case_when(
      jour >= bornes$p1[1] & jour <= bornes$p1[2] ~ "1_Artisanal",
      jour >= bornes$p2[1] & jour <= bornes$p2[2] ~ "2_Semi-pro",
      jour >= bornes$p3[1] & jour <= bornes$p3[2] ~ "3_Institutionnel",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(phase))

message(nrow(df_raw), " tokens après filtre POS + phases")

# ---------------------------------------------------------------------------
# HELPER : construire DTM text2vec
# ---------------------------------------------------------------------------

make_token_list <- function(df, doc_col) {
  df %>%
    group_by({{ doc_col }}) %>%
    summarise(tokens = list(lemma), doc_id = as.character(first({{ doc_col }})),
              .groups = "drop") %>%
    filter(lengths(tokens) >= MIN_TOKENS)
}

build_dtm_t2v <- function(tok_df) {
  it     <- itoken(tok_df$tokens, ids = tok_df$doc_id, progressbar = FALSE)
  vocab  <- create_vocabulary(it)
  vocab  <- prune_vocabulary(vocab, term_count_min = MIN_TF)
  vect   <- vocab_vectorizer(vocab)
  dtm    <- create_dtm(it, vect, type = "dgCMatrix")
  list(dtm = dtm, vocab = vocab)
}

# ---------------------------------------------------------------------------
# HELPER : lancer LDA text2vec et extraire beta + gamma (format tidy)
# ---------------------------------------------------------------------------

run_lda_t2v <- function(dtm, label = "") {
  set.seed(SEED)
  message(sprintf("  LDA text2vec — K=%d, %d docs, %d termes, iter=%d %s",
                  K, nrow(dtm), ncol(dtm), N_ITER, label))
  lda <- LDA$new(
    n_topics         = K,
    doc_topic_prior  = 50 / K,   # alpha symmétrique
    topic_word_prior = 0.01       # beta/eta
  )
  doc_topic <- lda$fit_transform(
    x = dtm, n_iter = N_ITER,
    convergence_tol = 1e-4, n_check_convergence = 50,
    progressbar = FALSE
  )
  list(lda = lda, doc_topic = doc_topic)
}

# beta tidy : K lignes × n_vocab  →  long (topic, term, beta)
# On évite pivot_longer sur les colnames (certains lemmes = noms invalides comme "...")
extract_beta <- function(lda_obj, vocab_terms) {
  tw <- lda_obj$topic_word_distribution    # K × n_vocab
  n_topics <- nrow(tw)
  n_terms  <- ncol(tw)
  data.frame(
    topic_num = rep(seq_len(n_topics), each = n_terms),
    term      = rep(vocab_terms, times = n_topics),
    beta      = as.vector(t(tw)),
    stringsAsFactors = FALSE
  )
}

# gamma tidy : n_docs × K  →  long (document, topic_num, gamma)
extract_gamma <- function(doc_topic_mat) {
  as.data.frame(doc_topic_mat) %>%
    mutate(document = rownames(doc_topic_mat)) %>%
    pivot_longer(-document, names_to = "topic_num", values_to = "gamma") %>%
    mutate(topic_num = as.integer(sub("V", "", topic_num)))
}

# ---------------------------------------------------------------------------
# 2A. LDA PAR MESSAGE
# ---------------------------------------------------------------------------

message("\n--- DTM par message ---")
tok_msg <- make_token_list(df_raw, message_id)
res_msg <- build_dtm_t2v(tok_msg)
fit_msg <- run_lda_t2v(res_msg$dtm, "(messages)")

beta_msg  <- extract_beta(fit_msg$lda, res_msg$vocab$term)
gamma_msg <- extract_gamma(fit_msg$doc_topic) %>%
  mutate(message_id = as.integer(document))

# Joindre dates
dates_msg <- df_raw %>% distinct(message_id, jour, mois, phase)
gamma_msg_dated <- gamma_msg %>% left_join(dates_msg, by = "message_id")

# ---------------------------------------------------------------------------
# 2B. LDA PAR MOIS
# ---------------------------------------------------------------------------

message("\n--- DTM par mois ---")
tok_mois <- df_raw %>%
  mutate(doc_mois = as.character(mois)) %>%
  make_token_list(doc_mois)

res_mois <- build_dtm_t2v(tok_mois)
fit_mois <- run_lda_t2v(res_mois$dtm, "(mois)")

beta_mois  <- extract_beta(fit_mois$lda, res_mois$vocab$term)
gamma_mois <- extract_gamma(fit_mois$doc_topic) %>%
  mutate(document = as.Date(document))

# ---------------------------------------------------------------------------
# 3. OUTPUT CONSOLE — top mots par topic
# ---------------------------------------------------------------------------

print_top_words <- function(beta_df, label = "MSG", n = 12) {
  message(sprintf("\n╔══════════════════════════════════════════════╗"))
  message(sprintf("║   TOP MOTS PAR TOPIC — modèle %-8s K=%d  ║", label, K))
  message(sprintf("╚══════════════════════════════════════════════╝"))
  top <- beta_df %>%
    group_by(topic_num) %>%
    slice_max(beta, n = n) %>%
    arrange(topic_num, desc(beta))
  for (t in sort(unique(top$topic_num))) {
    words <- top %>% filter(topic_num == t) %>% pull(term)
    message(sprintf("\n  Topic %d : %s", t, paste(words, collapse = " | ")))
  }
}

print_top_words(beta_msg,  "MSG")
print_top_words(beta_mois, "MOIS")

# ---------------------------------------------------------------------------
# 4. GRAPHIQUE BETA — top mots par topic
# ---------------------------------------------------------------------------

make_beta_plot <- function(beta_df, title_suffix) {
  top_words <- beta_df %>%
    group_by(topic_num) %>%
    slice_max(beta, n = 12) %>%
    ungroup() %>%
    mutate(
      topic_label = factor(paste0("Topic ", topic_num)),
      term        = tidytext::reorder_within(term, beta, topic_num)
    )

  ggplot(top_words, aes(x = term, y = beta, fill = factor(topic_num))) +
    geom_col(show.legend = FALSE, width = 0.75) +
    coord_flip() +
    facet_wrap(~ topic_label, scales = "free_y", ncol = 3) +
    tidytext::scale_x_reordered() +
    scale_fill_manual(values = TOPIC_COLORS) +
    scale_y_continuous(
      labels = scales::label_scientific(digits = 2),
      expand = expansion(mult = c(0, 0.1))
    ) +
    labs(
      title    = paste0("LDA K=6 — Mots caractéristiques par topic (β) — ", title_suffix),
      subtitle = sprintf(
        "Top 12 termes | text2vec VB %d iter | POS : NOUN, VERB, ADJ, PRON, ADV",
        N_ITER),
      x        = NULL,
      y        = "Probabilité β (mot | topic)",
      caption  = paste0(
        "β = probabilité qu'un mot soit tiré d'un topic donné.\n",
        "Étiquettes provisoires (Topic 1…6) — à renommer après inspection des mots caractéristiques."
      )
    ) +
    theme_madyar() +
    theme(
      strip.text = element_text(size = 9, face = "bold"),
      axis.text.y = element_text(size = 8),
      panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.3),
      panel.grid.major.y = element_blank()
    )
}

p_beta_msg  <- make_beta_plot(beta_msg,  "par message")
p_beta_mois <- make_beta_plot(beta_mois, "par mois")

save_plot(p_beta_msg,  file.path(OUT, "34a_lda_beta_messages.png"),
          format = "square", width = 13, dpi = 600)
save_plot(p_beta_mois, file.path(OUT, "34b_lda_beta_mois.png"),
          format = "square", width = 13, dpi = 600)

# ---------------------------------------------------------------------------
# 5. GAMMA TEMPOREL — stacked area mensuel (depuis modèle message)
# ---------------------------------------------------------------------------

gamma_mois_agg <- gamma_msg_dated %>%
  filter(!is.na(mois)) %>%
  group_by(mois, topic_num) %>%
  summarise(gamma_moy = mean(gamma), .groups = "drop") %>%
  group_by(mois) %>%
  mutate(gamma_norm = gamma_moy / sum(gamma_moy)) %>%
  ungroup() %>%
  mutate(topic_label = factor(paste0("Topic ", topic_num),
                               levels = paste0("Topic ", 1:K)))

names(TOPIC_COLORS) <- paste0("Topic ", 1:K)

p_gamma_msg <- ggplot(gamma_mois_agg,
                      aes(x = mois, y = gamma_norm, fill = topic_label)) +
  geom_area(position = "stack", alpha = 0.88, colour = "white", linewidth = 0.15) +
  geom_phase_lines() +
  scale_fill_manual(values = TOPIC_COLORS) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b\n%Y",
               expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Évolution mensuelle des topics LDA — modèle par message",
    subtitle = "γ moyen par mois normalisé | K=6 | text2vec Variational Bayes",
    x        = NULL,
    y        = "Part du topic dans le mois",
    fill     = NULL,
    caption  = paste0(
      "γ = probabilité d'appartenance d'un message à un topic (agrégé par mois, normalisé).\n",
      "Tirets = ruptures de phase (janv. 2024 / oct. 2024). Étiquettes à renommer."
    )
  ) +
  theme_madyar() +
  theme(legend.position = "bottom")

save_plot(p_gamma_msg, file.path(OUT, "34c_lda_gamma_temporal_messages.png"),
          format = "wide_16_9", width = 13, dpi = 600)

# ---------------------------------------------------------------------------
# 6. GAMMA TEMPOREL — modèle par mois (direct)
# ---------------------------------------------------------------------------

gamma_mois_long <- gamma_mois %>%
  mutate(topic_label = factor(paste0("Topic ", topic_num),
                               levels = paste0("Topic ", 1:K)))

p_gamma_mois <- ggplot(gamma_mois_long,
                       aes(x = document, y = gamma, fill = topic_label)) +
  geom_area(position = "stack", alpha = 0.88, colour = "white", linewidth = 0.15) +
  geom_phase_lines() +
  scale_fill_manual(values = TOPIC_COLORS) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b\n%Y",
               expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Évolution mensuelle des topics LDA — modèle par mois",
    subtitle = "Chaque mois = un document | K=6 | text2vec Variational Bayes",
    x        = NULL,
    y        = "γ (proportion du mois)",
    fill     = NULL,
    caption  = "Modèle entraîné avec un document = un mois (plus robuste, moins granulaire).\nTirets = ruptures de phase."
  ) +
  theme_madyar() +
  theme(legend.position = "bottom")

save_plot(p_gamma_mois, file.path(OUT, "34d_lda_gamma_temporal_mois.png"),
          format = "wide_16_9", width = 13, dpi = 600)

# ---------------------------------------------------------------------------
# 7. GAMMA PAR PHASE — violin + boxplot
# ---------------------------------------------------------------------------

gamma_phase <- gamma_msg_dated %>%
  filter(!is.na(phase)) %>%
  mutate(
    topic_label = factor(paste0("Topic ", topic_num),
                         levels = paste0("Topic ", 1:K)),
    phase = factor(phase, levels = c("1_Artisanal", "2_Semi-pro", "3_Institutionnel"))
  )

p_gamma_phase <- ggplot(gamma_phase,
                        aes(x = phase, y = gamma, fill = phase)) +
  geom_violin(alpha = 0.55, colour = NA, trim = TRUE) +
  geom_boxplot(width = 0.18, outlier.size = 0.4, outlier.alpha = 0.3,
               colour = "grey20", fill = "white", alpha = 0.85) +
  facet_wrap(~ topic_label, ncol = 3, scales = "free_y") +
  scale_fill_phase(short = TRUE) +
  scale_x_discrete(labels = c("Artisanal", "Semi-pro", "Institutionnel")) +
  labs(
    title    = "Distribution du γ par topic et par phase",
    subtitle = "Violin + boxplot | γ = probabilité d'appartenance au topic",
    x        = NULL,
    y        = "γ (probabilité topic | message)",
    caption  = "Un point = un message. Topic dominant dans une phase = distribution γ décalée vers le haut."
  ) +
  theme_madyar() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8, angle = 20, hjust = 1),
    strip.text  = element_text(size = 9)
  )

save_plot(p_gamma_phase, file.path(OUT, "34e_lda_gamma_par_phase.png"),
          format = "square", width = 12, dpi = 600)

# ---------------------------------------------------------------------------
# 8. EXPORT CSV
# ---------------------------------------------------------------------------

beta_msg %>%
  group_by(topic_num) %>% slice_max(beta, n = 20) %>%
  write_csv(file.path(OUT, "34_lda_beta_messages_top20.csv"))

beta_mois %>%
  group_by(topic_num) %>% slice_max(beta, n = 20) %>%
  write_csv(file.path(OUT, "34_lda_beta_mois_top20.csv"))

gamma_msg_dated %>%
  write_csv(file.path(OUT, "34_lda_gamma_messages.csv"))

gamma_mois %>%
  write_csv(file.path(OUT, "34_lda_gamma_mois.csv"))

message("\n✔ 5 PNG + 4 CSV dans ", OUT)
message("=== Terminé 34_lda_topics.R ===")
