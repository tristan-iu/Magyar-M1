# 17_crossposts.R — Crossposts par plateforme + liens de collecte
# Produit : 4_data_et_viz/17_crossposts_global.png
#           4_data_et_viz/17_crossposts_taux.png
#           4_data_et_viz/17_crossposts_global_solo.png
#           4_data_et_viz/17_crossposts_taux_solo.png
#           4_data_et_viz/17_crossposts_phases.png
#           4_data_et_viz/17_crossposts_collecte.png
#           4_data_et_viz/17_crossposts_collecte_phase.png
#           4_data_et_viz/17_crossposts_collecte_annee.png
#           4_data_et_viz/17_crossposts_cpt.png
#           4_data_et_viz/17_crossposts_mention_taux.png
#           4_data_et_viz/17_crossposts_mention_solo.png
#           4_data_et_viz/17_crossposts_collecte_detail.png
#           4_data_et_viz/17_crossposts_collecte_detail_stack.png
# Rscript 3b_stats_R/scripts_r/17_crossposts.R
#
# Source des données : champ `liens_externes` (liste JSON par message),
# enrichi en mai 2026 via enrich_liens_externes.py (Telethon entities).
# L'ancienne approche str_detect(legende, "youtube.com") ne capturait pas
# les liens inline (texte hyperlié ≠ URL visible) — désormais tous capturés.
# Note : Twitter/X = 1 seul lien sur tout le corpus (msg #349, jan. 2023)
# → non représenté.

this_file <- local({
  # 1. Rscript : argument --file=
  f <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  if (length(f) > 0 && nzchar(f)) return(normalizePath(f, mustWork = FALSE))
  # 2. source() dans un autre script : env$ofile
  for (env in rev(sys.frames()))
    if (!is.null(env$ofile)) return(normalizePath(env$ofile, mustWork = FALSE))
  # 3. RStudio interactif : rstudioapi (toujours disponible dans RStudio)
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      tryCatch(rstudioapi::isAvailable(), error = function(e) FALSE)) {
    ctx <- tryCatch(rstudioapi::getSourceEditorContext(), error = function(e) NULL)
    if (!is.null(ctx) && nzchar(ctx$path))
      return(normalizePath(ctx$path, mustWork = FALSE))
  }
  stop("Lancer via Rscript, source(), ou depuis RStudio avec le script actif ouvert.")
})
BASE <- dirname(dirname(this_file))
source(file.path(BASE, "r_source.R"))
OUT  <- file.path(BASE, "..", "4_data_et_viz")
dir.create(OUT, showWarnings = FALSE)
message("=== 17_crossposts.R — ", nrow(df_clean), " messages chargés ===")

# ---------------------------------------------------------------------------
# CONSTRUCTION DU DATA.FRAME DÉNORMALISÉ — un lien par ligne
# ---------------------------------------------------------------------------

# On s'assure que le champ est présent (colonne liste vide par défaut)
if (!"liens_externes" %in% names(df_clean)) {
  stop("Champ `liens_externes` absent de df_clean. Relancer enrich_liens_externes.py.")
}

# On classe chaque URL dans une catégorie analytique.
# Ordre case_when = du plus spécifique au plus général.
# .army = TLD ukrainien des unités militaires (magyarbirds.army, sbs-group.army…)
#
# Note encodage : les URLs brutes (MessageEntityUrl) peuvent être tronquées
# de 1-4 chars si des emojis précèdent l'URL dans le message (bug UTF-16
# corrigé dans enrich_liens_externes.py v2, mai 2026). Les patterns ci-dessous
# sont écrits pour matcher aussi les formes partielles courantes :
#   youtube.com → parfois "utube.com" ou "tube.com" (après fix: plus nécessaire)
#   https://    → parfois "tps://" ou "ps://"      (le domaine reste intact)
# Les liens inline (MessageEntityTextUrl) ne sont pas affectés car l'URL
# est dans ent.url, pas extraite du texte.
classifier_url <- function(url) {
  u <- tolower(url)
  dplyr::case_when(
    # YouTube — youtu.be et youtube.com (+ variantes tronquées pré-fix)
    stringr::str_detect(u, "youtu\\.be|youtube\\.com|utube\\.com")        ~ "YouTube",
    # Facebook — facebook.com, fb.watch (fb.com = 0 dans le corpus)
    stringr::str_detect(u, "facebook\\.com|fb\\.watch")                   ~ "Facebook",
    # Instagram — instagram.com (instagr.am = 0)
    stringr::str_detect(u, "instagram\\.com")                             ~ "Instagram",
    # TikTok
    stringr::str_detect(u, "tiktok\\.com")                                ~ "TikTok",
    # Sites officiels brigade — .army TLD ukrainien (magyarbirds.army, sbs-group.army…)
    # Pattern strict : ne pas matcher "magyarbirds" seul (subset de youtube.com/@magyarbirds)
    stringr::str_detect(u, "\\.army|magyarbirds\\.army|sbs-group\\.army") ~ "Site officiel (https://sbs-group.army/)",
    # Collecte — plateformes de fundraising avec URL
    # PrivatBank est absent (carte en texte brut) — détecté séparément via legende
    stringr::str_detect(u, "monobank|buymeacoffee|whitepay|forms\\.gle|dignitas\\.fund|marikatefund") ~ "Sites de collecte (MonoBank, BuyMeACoffee, etc.)",
    # Telegram — liens vers d'autres canaux/posts
    stringr::str_detect(u, "t\\.me/")                                     ~ "Telegram",
    TRUE                                                                   ~ NA_character_
  )
}

# On dénormalise : une ligne par (message × lien), en gardant les métadonnées
# utiles pour les analyses temporelles et par phase.
df_liens <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  select(message_id, mois, phase, legende, liens_externes) %>%
  # On exclut les messages sans liens (liste vide ou NULL)
  filter(map_int(liens_externes, length) > 0) %>%
  unnest_longer(liens_externes, values_to = "lien") %>%
  unnest_wider(lien) %>%                    # colonnes : url, texte
  filter(!is.na(url), nchar(url) > 5) %>%
  mutate(categorie = classifier_url(url)) %>%
  filter(!is.na(categorie))                 # on écarte les URLs non classifiées

# PrivatBank — numéros de carte en texte brut (pas de URL, pas dans liens_externes)
# On fabrique des "liens virtuels" pour uniformiser l'analyse collecte.
# Détection : "приватбанк" ou "privatbank" dans la légende (hors messages
# qui ont déjà un lien Collecte pour éviter le double-comptage).
# Constantes — labels internes issus de classifier_url (longs mais exacts)
CAT_BRIGADE  <- "Site officiel (https://sbs-group.army/)"
CAT_COLLECTE <- "Sites de collecte (MonoBank, BuyMeACoffee, etc.)"

# Labels courts pour les légendes de graphiques
LBL_CATS_COURT <- c(
  "YouTube"    = "YouTube",
  "Facebook"   = "Facebook",
  "Instagram"  = "Instagram",
  "TikTok"     = "TikTok",
  "Telegram"   = "Telegram",
  "Aucun lien" = "Aucun lien"
)
LBL_CATS_COURT[[CAT_BRIGADE]]  <- "Site brigade"
LBL_CATS_COURT[[CAT_COLLECTE]] <- "Collecte"

ids_deja_collecte <- df_liens %>%
  filter(categorie == CAT_COLLECTE) %>%
  distinct(message_id) %>%
  pull(message_id)

df_privat <- df_clean %>%
  filter(!is.na(date), !is.na(legende)) %>%
  filter(str_detect(tolower(legende), "приватбанк|privatbank")) %>%
  filter(!message_id %in% ids_deja_collecte) %>%
  mutate(
    mois      = as.Date(floor_date(date, "month")),
    url       = "privatbank-text",
    texte     = NA_character_,
    categorie = CAT_COLLECTE
  ) %>%
  select(message_id, mois, phase, legende, url, texte, categorie)

# On unifie les deux sources de collecte
df_liens <- bind_rows(df_liens, df_privat)

n_liens <- nrow(df_liens)
n_msg   <- n_distinct(df_liens$message_id)
message(sprintf("  %d liens classifiés sur %d messages", n_liens, n_msg))
message("  Répartition :")
df_liens %>% count(categorie, sort = TRUE) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  { message(capture.output(print(.))); . }

# ---------------------------------------------------------------------------
# DONNÉES DE BASE — série mensuelle par catégorie
# ---------------------------------------------------------------------------
# On compte les MESSAGES distincts avec au moins 1 lien vers chaque catégorie
# (pas le total de liens) — évite de surpondérer les messages multi-liens.
# On ajoute la phase dès maintenant — réutilisée dans plusieurs figures.

CATS_PRINCIPALES <- c("YouTube", "Facebook", "Instagram",
                      "Telegram", CAT_BRIGADE, CAT_COLLECTE)

df_plot <- df_liens %>%
  filter(categorie %in% CATS_PRINCIPALES) %>%
  distinct(mois, message_id, categorie) %>%         # 1 ligne par (message, catégorie)
  count(mois, categorie, name = "n") %>%
  complete(
    mois      = seq(min(df_liens$mois), max(df_liens$mois), by = "month"),
    categorie = CATS_PRINCIPALES,
    fill      = list(n = 0)
  ) %>%
  mutate(
    categorie = factor(categorie, levels = CATS_PRINCIPALES),
    phase = case_when(
      mois >= bornes$p1[1] & mois <= bornes$p1[2] ~ 1L,
      mois >= bornes$p2[1] & mois <= bornes$p2[2] ~ 2L,
      mois >= bornes$p3[1] & mois <= bornes$p3[2] ~ 3L,
      TRUE ~ NA_integer_
    )
  )

# On associe une couleur à chaque catégorie (order = PAL_CAT, stable pour toutes les figures)
noms_cat <- setNames(PAL_CAT[seq_along(CATS_PRINCIPALES)], CATS_PRINCIPALES)

# Ordre d'empilement pour les barres : Site brigade en bas (croissance P3
# visible depuis la baseline = 0) et Collecte en haut (disparition P3
# visible au sommet). Le milieu regroupe les plateformes présentes tout au
# long du corpus. TikTok (4 liens sur 37 mois) est exclu — invisible en stack.
CATS_STACK <- c(CAT_BRIGADE, "Telegram", "YouTube", "Facebook", "Instagram", CAT_COLLECTE)

df_stack <- df_plot %>%
  filter(categorie %in% CATS_STACK) %>%
  mutate(categorie = factor(categorie, levels = CATS_STACK))

# ---------------------------------------------------------------------------
# FIGURE 1 — Évolution mensuelle par catégorie (barres empilées, absolu)
# ---------------------------------------------------------------------------

p_global <- ggplot(df_stack, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27) +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_cat, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  geom_phase_lines() +
  labs(
    title    = "Crossposts par plateforme — évolution mensuelle",
    subtitle = "Somme des messages distincts avec au moins un lien vers chaque service (un message peut compter dans plusieurs catégories).\nSource : entités Telegram (liens inline + URLs brutes).",
    x = NULL, y = "Messages avec lien (empilés)"
  )

save_plot(p_global,
          file.path(OUT, "17_crossposts_global.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 2 — Composition mensuelle normalisée (barres 100% empilées)
# ---------------------------------------------------------------------------
# Chaque barre = 100% de l'activité crosspost de ce mois.
# Normalisation interne au mix de liens (≠ % du total messages du mois).

p_taux <- ggplot(df_stack, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27, position = "fill") +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_cat, name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  geom_phase_lines() +
  labs(
    title    = "Crossposts par plateforme — composition mensuelle (100%)",
    subtitle = "Part de chaque plateforme dans le mix de liens du mois.\nSite brigade \u2191 en P3, Collecte \u2193 en P3.",
    x = NULL, y = "Part du mix crosspost"
  )

save_plot(p_taux,
          file.path(OUT, "17_crossposts_taux.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 1b — Global avec "Aucun lien" comme 7ème catégorie
# ---------------------------------------------------------------------------
# Identique à Figure 1 (même format barres empilées), mais on ajoute un
# segment gris au sommet représentant les messages sans lien classifié.
# "Aucun" = n_total - messages_distincts_avec_lien : pas de double-comptage
# (un message a un lien ou n'en a pas, sans ambiguïté). Les 6 catégories
# colorées peuvent se chevaucher (un message YouTube peut aussi avoir Facebook).

df_avec_lien_mois <- df_liens %>%
  distinct(mois, message_id) %>%
  count(mois, name = "n_avec")

df_total_mois <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  count(mois, name = "n_total")

df_aucun_mois <- df_total_mois %>%
  left_join(df_avec_lien_mois, by = "mois") %>%
  mutate(
    n_avec    = replace_na(n_avec, 0L),
    n         = pmax(0L, as.integer(n_total - n_avec)),
    categorie = "Aucun lien"
  ) %>%
  select(mois, categorie, n)

# "Aucun lien" en dernier = au sommet de la pile
CATS_AVEC_AUCUN <- c(CATS_STACK, "Aucun lien")
noms_cat_aucun  <- c(noms_cat[CATS_STACK], "Aucun lien" = "grey80")

df_stack_aucun <- df_stack %>%
  select(mois, categorie, n) %>%
  mutate(categorie = as.character(categorie)) %>%
  bind_rows(df_aucun_mois) %>%
  mutate(categorie = factor(categorie, levels = CATS_AVEC_AUCUN))

p_global_solo <- ggplot(df_stack_aucun, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27) +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_cat_aucun, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  geom_phase_lines() +
  labs(
    title    = "Crossposts par plateforme + messages sans lien (gris)",
    subtitle = "Segment gris = messages sans lien vers aucune plateforme connue.\nLes segments colorés peuvent compter le même message dans plusieurs catégories.",
    x = NULL, y = "Messages publiés"
  )

save_plot(p_global_solo,
          file.path(OUT, "17_crossposts_global_solo.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 2b — Composition 100% avec "Aucun lien"
# ---------------------------------------------------------------------------
# Même format que Figure 2 (position = "fill"), avec "Aucun" en gris au sommet.
# La normalisation porte sur sum(toutes catégories + Aucun) — ce qui signifie
# que la part de chaque catégorie colorée est légèrement sous-estimée dans
# les mois avec beaucoup de multi-liens, mais "Aucun" reste lisible comme
# signal de montée/baisse du crossposting.

p_taux_solo <- ggplot(df_stack_aucun, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27, position = "fill") +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_cat_aucun, name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  geom_phase_lines() +
  labs(
    title    = "Vers où publie-t-on ?",
    subtitle = "Composition mensuelle des messages avec liens.",
    x = NULL, y = "Pourcentage de publications"
  )

save_plot(p_taux_solo,
          file.path(OUT, "17_crossposts_taux_solo.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 3 — Crossposts par phase (P1 / P2 / P3 séparés)
# ---------------------------------------------------------------------------
# Un panneau par phase avec son propre axe X — permet de zoomer sur la
# dynamique interne de chaque période sans que les phases longues écrasent
# les courtes.

LBL_PHASES_PANEL <- c(
  "1" = "P1 — Artisanal (sept. 2022 \u2013 déc. 2023)",
  "2" = "P2 — Semi-pro (janv. \u2013 sept. 2024)",
  "3" = "P3 — Institutionnel (oct. 2024 \u2013 sept. 2025)"
)

df_plot_phases <- df_plot %>%
  filter(!is.na(phase)) %>%
  mutate(
    phase_label = factor(
      LBL_PHASES_PANEL[as.character(phase)],
      levels = unname(LBL_PHASES_PANEL)   # garantit l'ordre P1 → P2 → P3
    )
  )

p_phases <- ggplot(df_plot_phases, aes(x = mois, y = n, colour = categorie)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  facet_wrap(~ phase_label, ncol = 1, scales = "free_x") +
  scale_x_date(
    date_breaks = "2 months", date_labels = "%b\n%Y",
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  scale_colour_manual(values = noms_cat, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Crossposts par plateforme — dynamique par phase",
    subtitle = "Messages distincts par mois. Un panneau par phase, axe temporel libre.",
    x = NULL, y = "Messages avec lien"
  )

save_plot(p_phases,
          file.path(OUT, "17_crossposts_phases.png"),
          format = "square", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 4 — Liens de collecte par phase
# ---------------------------------------------------------------------------
# monobank, BuyMeACoffee, WhitePay, Google Forms sont des plateformes de
# fundraising ou de commandes groupées d'équipement.

df_collecte <- df_liens %>%
  filter(categorie == CAT_COLLECTE) %>%
  distinct(message_id, mois, phase)

# On recompte par mois pour la série temporelle
df_collecte_mois <- df_collecte %>%
  count(mois, name = "n") %>%
  complete(
    mois = seq(min(df_liens$mois), max(df_liens$mois), by = "month"),
    fill = list(n = 0)
  )

# Totaux par phase pour annotation
df_collecte_phase <- df_collecte %>%
  filter(!is.na(phase)) %>%
  count(phase, name = "n_msg") %>%
  mutate(label = paste0(n_msg, " msg."))

# Série temporelle avec barres
p_collecte <- ggplot(df_collecte_mois, aes(x = mois, y = n)) +
  geom_col(fill = PAL_CAT[5], alpha = 0.85, width = 25) +
  geom_phase_lines() +
  scale_x_mois(breaks = "3 months") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title    = "Liens de collecte de fonds — évolution mensuelle",
    subtitle = "Plateformes : Monobank \u00b7 BuyMeACoffee \u00b7 WhitePay \u00b7 Google Forms.\nSignal H2 : disparition du fundraising personnel en phase institutionnelle (P3).",
    x = NULL, y = "Messages avec lien de collecte"
  )

# On annote les totaux par phase en un seul appel annotate vectorisé.
# Un appel par phase (boucle) conflicte avec l'annotate interne de
# geom_phase_lines() — ggplot2 mélange les longueurs de données entre layers.
milieux_phase <- as.Date(c("2023-03-01", "2024-05-01", "2025-03-01"))
labels_phase  <- c("P1", "P2", "P3")
y_annot       <- max(df_collecte_mois$n, na.rm = TRUE) * 0.88

totaux_phase <- sapply(1:3, function(ph) {
  n <- df_collecte_phase %>% filter(phase == ph) %>% pull(n_msg)
  if (length(n) == 0) 0L else n
})

# geom_text avec data explicite — évite le conflit de longueur entre
# la data du plot (n lignes) et les 3 points d'annotation quand annotate()
# hérite de la data principale du ggplot.
df_annot_phase <- tibble(
  mois  = milieux_phase,
  y     = y_annot,
  label = paste0(labels_phase, "\n", totaux_phase, " msg.")
)

p_collecte <- p_collecte +
  geom_text(data = df_annot_phase,
            aes(x = mois, y = y, label = label),
            size = 3.2, colour = "grey30", hjust = 0.5,
            inherit.aes = FALSE)

save_plot(p_collecte,
          file.path(OUT, "17_crossposts_collecte.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 4b — Collecte par phase (vue synthétique)
# ---------------------------------------------------------------------------
# Durée de chaque phase en mois — pour annoter avec le taux mensuel moyen
# et rendre les phases comparables malgré leurs durées inégales (P1=16 mois,
# P2=9 mois, P3=12 mois).

n_mois_phase <- tibble(
  phase  = c(1L, 2L, 3L),
  n_mois = c(
    length(seq(bornes$p1[1], bornes$p1[2], by = "month")),
    length(seq(bornes$p2[1], bornes$p2[2], by = "month")),
    length(seq(bornes$p3[1], bornes$p3[2], by = "month"))
  )
)

lbl_phase_bar <- c("1" = "P1\nArtisanal",
                   "2" = "P2\nSemi-pro",
                   "3" = "P3\nInstit.")
pal_phase_num <- c(
  "1" = unname(PAL_PHASE["1_Artisanal"]),
  "2" = unname(PAL_PHASE["2_Semi-pro"]),
  "3" = unname(PAL_PHASE["3_Institutionnel"])
)

# Total de messages publiés par phase — dénominateur pour le taux
df_total_phase <- df_clean %>%
  filter(!is.na(phase)) %>%
  count(phase, name = "n_total_phase")

df_cp <- df_collecte_phase %>%
  left_join(df_total_phase, by = "phase") %>%
  mutate(
    pct   = round(100 * n_msg / n_total_phase, 1),
    annot = paste0(pct, "%\n(", n_msg, " / ", n_total_phase, ")")
  )

p_collecte_phase <- ggplot(
  df_cp,
  aes(x = factor(phase), y = pct, fill = factor(phase))
) +
  geom_col(width = 0.6) +
  geom_text(aes(label = annot), vjust = -0.35, size = 3.5, colour = "grey20") +
  scale_x_discrete(labels = lbl_phase_bar) +
  scale_fill_manual(values = pal_phase_num, guide = "none") +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.22)),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title    = "Liens de collecte de fonds — taux par phase",
    subtitle = "% des messages de la phase contenant au moins un lien de collecte.\nEntre parenthèses : n messages avec lien / total messages de la phase.",
    x = NULL, y = "% des messages de la phase"
  )

save_plot(p_collecte_phase,
          file.path(OUT, "17_crossposts_collecte_phase.png"),
          format = "wide_16_9", width = 8, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 4c — Collecte par année civile (taux)
# ---------------------------------------------------------------------------
# % des messages publiés dans l'année avec au moins un lien de collecte.
# Corrige le biais des années partielles (2022 = 4 mois, 2025 = 9 mois).

df_total_annee <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(annee = lubridate::year(date)) %>%
  count(annee, name = "n_total_annee")

df_collecte_annee <- df_collecte %>%
  mutate(annee = lubridate::year(mois)) %>%
  count(annee, name = "n_msg") %>%
  left_join(df_total_annee, by = "annee") %>%
  mutate(
    pct   = round(100 * n_msg / n_total_annee, 1),
    annot = case_when(
      annee == 2022 ~ paste0(pct, "%\n(4 mois)"),
      annee == 2025 ~ paste0(pct, "%\n(9 mois)"),
      TRUE          ~ paste0(pct, "%\n(", n_msg, " / ", n_total_annee, ")")
    )
  )

p_collecte_annee <- ggplot(
  df_collecte_annee,
  aes(x = factor(annee), y = pct)
) +
  geom_col(fill = PAL_CAT[5], alpha = 0.85, width = 0.6) +
  geom_text(aes(label = annot), vjust = -0.35, size = 3.5, colour = "grey20") +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.2)),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title    = "Liens de collecte de fonds — taux par année civile",
    subtitle = "% des messages publiés dans l'année avec au moins un lien de collecte.\n* 2022 = sept.\u2013déc. uniquement ; 2025 = janv.\u2013sept. uniquement.",
    x = NULL, y = "% des messages de l'année"
  )

save_plot(p_collecte_annee,
          file.path(OUT, "17_crossposts_collecte_annee.png"),
          format = "wide_16_9", width = 8, dpi = 300)

# ---------------------------------------------------------------------------
# FIGURE 5 — Changepoints par plateforme (facets)
# ---------------------------------------------------------------------------
# Un changepoint global sur le total mélange des signaux opposés (collecte
# descend, site brigade monte). On calcule PELT indépendamment pour chaque
# plateforme et on facète — chaque panneau a sa propre échelle Y.

df_cpts <- CATS_PRINCIPALES %>%
  purrr::set_names() %>%
  purrr::map(function(cat) {
    d <- df_plot %>% filter(categorie == cat)
    dates <- compute_cpts(d$n, d$mois)
    if (length(dates) == 0) return(NULL)
    tibble(categorie = cat, mois_cpt = dates)
  }) %>%
  purrr::compact() %>%
  dplyr::bind_rows() %>%
  mutate(categorie = factor(categorie, levels = CATS_PRINCIPALES))

p_cpt <- ggplot(df_plot, aes(x = mois, y = n, colour = categorie)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  # Lignes de phase : geom_vline direct (pas geom_phase_lines) — l'annotate
  # texte de geom_phase_lines() plante en facet_wrap car ggplot2 tente
  # d'étendre 2 labels × 6 panneaux = 12 lignes, ce qui dépasse la longueur.
  geom_vline(xintercept = as.numeric(PHASE_DATES),
             colour = "#888888", linetype = "dashed", linewidth = 0.5) +
  { if (nrow(df_cpts) > 0)
      geom_vline(data = df_cpts,
                 aes(xintercept = as.numeric(mois_cpt)),
                 colour = "firebrick", linetype = "dashed",
                 linewidth = 0.6, inherit.aes = FALSE)
  } +
  facet_wrap(~ categorie, ncol = 2, scales = "free_y") +
  scale_x_mois(breaks = "6 months") +
  scale_colour_manual(values = noms_cat, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Crossposts — ruptures par plateforme (PELT)",
    subtitle = "Messages distincts par mois, échelle libre par panneau. Ligne rouge = rupture de moyenne/variance (PELT-MBIC).",
    x = NULL, y = "Messages avec lien"
  )

save_plot(p_cpt,
          file.path(OUT, "17_crossposts_cpt.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# ---------------------------------------------------------------------------
# NETTOYAGE — fichiers obsolètes
# ---------------------------------------------------------------------------
# On supprime les anciens outputs dont les noms ont changé pour éviter
# de confondre des viz stale avec les nouvelles lors de la rédaction.

for (f_old in c("17_crossposts_global_cpt.png", "30_crossposts_p1_cpt.png")) {
  p_old <- file.path(OUT, f_old)
  if (file.exists(p_old)) {
    file.remove(p_old)
    message("  Supprimé (obsolète) : ", f_old)
  }
}

# ---------------------------------------------------------------------------
# RÉCAPITULATIF CONSOLE
# ---------------------------------------------------------------------------

message("\n=== Récapitulatif ===")
message(sprintf("Messages avec au moins 1 lien externe : %d / %d (%.0f%%)",
  n_msg, nrow(df_clean), 100 * n_msg / nrow(df_clean)))

message("\nTop catégories (messages distincts par phase) :")
df_liens %>%
  filter(categorie %in% CATS_PRINCIPALES) %>%
  distinct(message_id, phase, categorie) %>%
  count(categorie, phase, name = "n") %>%
  tidyr::pivot_wider(names_from = phase, values_from = n, values_fill = 0) %>%
  { message(capture.output(print(.))); . }

message("\nCollecte par phase :")
df_collecte_phase %>% { message(capture.output(print(.))); . }

message(sprintf("\nSite brigade (.army) : %d liens sur %d messages",
  sum(df_liens$categorie == CAT_BRIGADE),
  n_distinct(df_liens$message_id[df_liens$categorie == CAT_BRIGADE])))

message(sprintf("\nChangepoints détectés : %d (sur %d plateformes)",
  nrow(df_cpts), n_distinct(df_cpts$categorie)))
if (nrow(df_cpts) > 0) {
  df_cpts %>%
    mutate(mois_cpt = format(mois_cpt, "%Y-%m")) %>%
    { message(capture.output(print(.))); . }
}

# ---------------------------------------------------------------------------
# FIGURE 6 — Mentions de plateformes dans légende + dialogue (regex)
# ---------------------------------------------------------------------------
# Complète l'analyse de liens (URLs cliquables) avec les mentions textuelles :
# un message peut citer "моноbank" ou "YouTube" sans créer de lien cliquable.
# Source : champs `legende` + `dialogue` (transcription Whisper, si disponible).
# Un message est compté UNE FOIS par catégorie même si mentionné plusieurs fois
# ou dans les deux champs.

MENTION_PATTERNS <- list(
  "YouTube"      = "youtube|youtu\\.be|\u044e\u0442\u0443\u0431",
  "Facebook"     = "facebook|fb\\.watch|\u0444\u0435\u0439\u0441\u0431\u0443\u043a|\u0444\u0435\u0441\u0431\u0443\u043a",
  "Instagram"    = "instagram|\u0456\u043d\u0441\u0442\u0430\u0433\u0440\u0430\u043c|\u0438\u043d\u0441\u0442\u0430\u0433\u0440\u0430\u043c",
  "TikTok"       = "tiktok|\u0442\u0456\u043a.\u0442\u043e\u043a|\u0442\u0438\u043a.\u0442\u043e\u043a",
  "Telegram"     = "telegram|\u0442\u0435\u043b\u0435\u0433\u0440\u0430\u043c|t\\.me/",
  "Site brigade" = "magyarbirds|\\.army|sbs.?group",
  "Collecte"     = "monobank|\u043c\u043e\u043d\u043e\u0431\u0430\u043d\u043a|buymeacoffee|whitepay|\u043f\u0440\u0438\u0432\u0430\u0442\u0431\u0430\u043d\u043a|privatbank|forms\\.gle|dignitas|marikatefund"
)

# On combine légende et dialogue en un seul texte par message (minuscules)
# dialogue peut être absent (NA) — replace_na("") pour éviter les NA dans paste()
df_texte <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(
    mois  = as.Date(floor_date(date, "month")),
    texte = tolower(paste(
      replace_na(legende,  ""),
      replace_na(dialogue, ""),
      sep = " "
    ))
  ) %>%
  select(message_id, mois, phase, texte)

# Pour chaque plateforme, on identifie les messages qui la mentionnent
df_mention_long <- purrr::map_dfr(
  names(MENTION_PATTERNS),
  function(cat) {
    df_texte %>%
      filter(str_detect(texte, MENTION_PATTERNS[[cat]])) %>%
      mutate(categorie = cat) %>%
      select(message_id, mois, phase, categorie)
  }
)

n_msg_mention <- n_distinct(df_mention_long$message_id)
message(sprintf("\n  Mentions texte : %d messages citent au moins une plateforme", n_msg_mention))
df_mention_long %>% count(categorie, sort = TRUE) %>%
  mutate(pct = round(100 * n / nrow(df_texte), 1)) %>%
  { message(capture.output(print(.))); . }

# Série mensuelle des mentions — mêmes catégories que les charts de liens
CATS_MENTION <- names(MENTION_PATTERNS)   # ordre = ordre MENTION_PATTERNS

# Palette cohérente avec les charts de liens (même couleur par plateforme
# dans les deux analyses)
noms_mention <- c(
  "Site brigade" = unname(PAL_CAT[1]),
  "Telegram"     = unname(PAL_CAT[2]),
  "YouTube"      = unname(PAL_CAT[3]),
  "Facebook"     = unname(PAL_CAT[4]),
  "Instagram"    = unname(PAL_CAT[5]),
  "Collecte"     = unname(PAL_CAT[6]),
  "TikTok"       = "grey60",
  "Aucun lien"   = "grey85"
)

df_plot_mention <- df_mention_long %>%
  distinct(mois, message_id, categorie) %>%
  count(mois, categorie, name = "n") %>%
  complete(
    mois      = seq(min(df_texte$mois), max(df_texte$mois), by = "month"),
    categorie = CATS_MENTION,
    fill      = list(n = 0)
  ) %>%
  mutate(categorie = factor(categorie, levels = CATS_MENTION))

# — Figure 6a : barres empilées 100% (composition des mentions)
CATS_MENTION_STACK <- c("Site brigade", "Telegram", "YouTube",
                        "Facebook", "Instagram", "Collecte", "TikTok")

df_mention_stack <- df_plot_mention %>%
  filter(categorie %in% CATS_MENTION_STACK) %>%
  mutate(categorie = factor(as.character(categorie), levels = CATS_MENTION_STACK))

p_mention_taux <- ggplot(df_mention_stack, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27, position = "fill") +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_mention, name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  geom_phase_lines() +
  labs(
    title    = "Mentions de plateformes (légende + dialogue) — composition mensuelle",
    subtitle = "Part de chaque plateforme dans les mentions textuelles du mois (legende + transcription Whisper).\nDétection par regex (fr/uk/ru). Un message compte une fois par plateforme même si cité plusieurs fois.",
    x = NULL, y = "Part du mix mentions"
  )

save_plot(p_mention_taux,
          file.path(OUT, "17_crossposts_mention_taux.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# — Figure 6b : même chart avec "Aucun" visible (messages sans mention du tout)
df_aucun_mention <- df_total_mois %>%
  left_join(
    df_mention_long %>% distinct(mois, message_id) %>% count(mois, name = "n_avec"),
    by = "mois"
  ) %>%
  mutate(
    n_avec    = replace_na(n_avec, 0L),
    n         = pmax(0L, as.integer(n_total - n_avec)),
    categorie = "Aucun lien"
  ) %>%
  select(mois, categorie, n)

CATS_MENTION_AUCUN <- c(CATS_MENTION_STACK, "Aucun lien")

df_mention_stack_aucun <- df_mention_stack %>%
  select(mois, categorie, n) %>%
  mutate(categorie = as.character(categorie)) %>%
  bind_rows(df_aucun_mention) %>%
  mutate(categorie = factor(categorie, levels = CATS_MENTION_AUCUN))

p_mention_solo <- ggplot(df_mention_stack_aucun, aes(x = mois, y = n, fill = categorie)) +
  geom_col(width = 27, position = "fill") +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = noms_mention, name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  geom_phase_lines() +
  labs(
    title    = "Mentions de plateformes + messages sans mention (gris)",
    subtitle = "Segment gris = messages sans mention d'aucune plateforme connue dans la légende ou le dialogue.",
    x = NULL, y = "Part du mix mensuel"
  )

save_plot(p_mention_solo,
          file.path(OUT, "17_crossposts_mention_solo.png"),
          format = "wide_16_9", width = 12, dpi = 300)



# ---------------------------------------------------------------------------
# FIGURE 7 — Collecte : détail par sous-plateforme (URL + mentions + IBAN)
# ---------------------------------------------------------------------------
# Chaque plateforme de collecte est analysée séparément.
# Sources combinées (dédupliquées par message × sous-plateforme) :
#   1. URLs classifiées dans df_liens (CAT_COLLECTE) — liens cliquables Telegram
#   2. Mentions textuelles dans legende + dialogue (df_texte, déjà construit)
#   3. IBAN ukrainien — ua[0-9]{25,27} après tolower() (numéros de virement)
#   4. Numéros de carte Oschadbank (préfixe 1020 — carte Visa state bank)
#   5. PayPal — email brovdi.edgar@yahoo.com (canal réel P1, 82 messages)
#   6. USDT TRC20 — wallet crypto (5 messages, même adresse sur toute la période)

# On sous-classifie les URLs déjà catégorisées comme CAT_COLLECTE
sous_classifier_collecte <- function(url) {
  u <- tolower(url)
  dplyr::case_when(
    str_detect(u, "monobank")       ~ "Monobank",
    str_detect(u, "buymeacoffee")   ~ "BuyMeACoffee",
    str_detect(u, "whitepay")       ~ "WhitePay",
    str_detect(u, "forms\\.gle")    ~ "Google Forms",
    str_detect(u, "dignitas")       ~ "Dignitas",
    str_detect(u, "marikatefund")   ~ "Marikatefund",
    u == "privatbank-text"          ~ "PrivatBank",
    TRUE                            ~ NA_character_
  )
}

df_sous_url <- df_liens %>%
  filter(categorie == CAT_COLLECTE) %>%
  mutate(sous_cat = sous_classifier_collecte(url)) %>%
  filter(!is.na(sous_cat)) %>%
  distinct(message_id, mois, phase, sous_cat)

# Patterns textuels par sous-plateforme (texte en minuscules = tolower appliqué)
# IBAN ukrainien     : "UA" + 25-27 chiffres → "ua" après tolower()
# Oschadbank         : préfixe carte 1020 + 12 chiffres (Visa state bank)
# PayPal             : email `brovdi.edgar` ou mot-clé paypal
# USDT TRC20         : wallet commençant par T + 33 alphanum (regex case-sensitive
#                      → on cherche dans le texte ORIGINAL, pas tolower)
SOUS_PATTERNS <- list(
  "Monobank"     = "monobank|\u043c\u043e\u043d\u043e\u0431\u0430\u043d\u043a",
  "PrivatBank"   = "\u043f\u0440\u0438\u0432\u0430\u0442\u0431\u0430\u043d\u043a|privatbank",
  "BuyMeACoffee" = "buymeacoffee",
  "WhitePay"     = "whitepay",
  "Google Forms" = "forms\\.gle",
  "PayPal"       = "paypal",
  "Oschadbank"   = "\u043e\u0449\u0430\u0434\u0431\u0430\u043d\u043a|oschadbank|1020[\\s]?[0-9]{4}[\\s]?[0-9]{4}[\\s]?[0-9]{4}",
  "IBAN"         = "ua[0-9]{25,27}",
  "Dignitas"     = "dignitas",
  "Marikatefund" = "marikatefund"
)

# USDT : wallets TRC20 (T + 33 alphanum) — on cherche dans le texte original
# (avant tolower) car les adresses crypto sont case-sensitive
df_usdt <- df_clean %>%
  filter(!is.na(date)) %>%
  mutate(mois = as.Date(floor_date(date, "month"))) %>%
  filter(
    str_detect(replace_na(legende, ""), "[T][A-Za-z0-9]{33}") |
    str_detect(replace_na(dialogue, ""), "[T][A-Za-z0-9]{33}")
  ) %>%
  mutate(sous_cat = "USDT (crypto)") %>%
  select(message_id, mois, phase, sous_cat)

# On détecte les mentions pour chaque sous-plateforme dans le texte combiné
# (df_texte construit en section Figure 6 — légende + dialogue, minuscules)
df_sous_texte <- purrr::map_dfr(names(SOUS_PATTERNS), function(cat) {
  df_texte %>%
    filter(str_detect(texte, SOUS_PATTERNS[[cat]])) %>%
    mutate(sous_cat = cat) %>%
    select(message_id, mois, phase, sous_cat)
})

# On fusionne les trois sources et on déduplique : 1 ligne par (message × sous-cat)
df_collecte_detail <- bind_rows(df_sous_url, df_sous_texte, df_usdt) %>%
  distinct(message_id, mois, phase, sous_cat)

n_corpus_total <- nrow(dplyr::filter(df_clean, !is.na(date)))
message(sprintf("\n--- Figure 7 : %d messages distincts avec sous-plateforme collecte ---",
  n_distinct(df_collecte_detail$message_id)))
df_collecte_detail %>% count(sous_cat, sort = TRUE) %>%
  mutate(pct_corpus = round(100 * n / n_corpus_total, 1)) %>%
  { message(capture.output(print(.))); . }

SOUS_CATS <- c("Monobank", "PrivatBank", "BuyMeACoffee", "WhitePay",
               "Google Forms", "PayPal", "Oschadbank", "IBAN",
               "USDT (crypto)", "Dignitas", "Marikatefund")

PAL_SOUS <- c(
  "Monobank"     = "#1565C0",
  "PrivatBank"   = "#4CAF50",
  "BuyMeACoffee" = "#F5A623",
  "WhitePay"     = "#9C27B0",
  "Google Forms" = "#EA4335",
  "PayPal"       = "#003087",
  "Oschadbank"   = "#006400",
  "IBAN"         = "#607D8B",
  "USDT (crypto)"= "#26A17B",
  "Dignitas"     = "#FF9800",
  "Marikatefund" = "#E91E63"
)

# Série mensuelle : % des messages du mois qui mentionnent chaque sous-plateforme
# df_total_mois déjà construit (section Figure 1b)
df_sous_mois <- df_collecte_detail %>%
  count(mois, sous_cat, name = "n_msg") %>%
  complete(
    mois     = seq(min(df_texte$mois), max(df_texte$mois), by = "month"),
    sous_cat = SOUS_CATS,
    fill     = list(n_msg = 0)
  ) %>%
  left_join(df_total_mois, by = "mois") %>%
  mutate(
    pct      = 100 * n_msg / replace_na(n_total, 1L),
    sous_cat = factor(sous_cat, levels = SOUS_CATS)
  )

# — Figure 7a : courbes (% mensuel par sous-plateforme)
p_collecte_detail <- ggplot(df_sous_mois, aes(x = mois, y = pct, colour = sous_cat)) +
  geom_line(linewidth = 0.9) +
  geom_point(data = dplyr::filter(df_sous_mois, n_msg > 0), size = 1.8) +
  scale_x_mois(breaks = "3 months") +
  scale_colour_manual(values = PAL_SOUS, name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.08)),
    labels = function(x) paste0(x, "%")
  ) +
  geom_phase_lines() +
  labs(
    title    = "Collecte — détail par sous-plateforme (% mensuel)",
    subtitle = paste0(
      "Sources : liens URL Telegram + mentions légende/dialogue + IBAN (UA+25-27 chiffres) + carte Oschadbank (1020…) + PayPal + wallet USDT TRC20.\n",
      "Un message peut figurer dans plusieurs sous-catégories. % = messages avec mention / total messages du mois."
    ),
    x = NULL, y = "% des messages du mois"
  )

save_plot(p_collecte_detail,
          file.path(OUT, "17_crossposts_collecte_detail.png"),
          format = "wide_16_9", width = 12, dpi = 300)

# — Figure 7b : barres empilées 100% (composition des mentions collecte par mois)
# On exclut Dignitas et Marikatefund (1 message chacun — invisibles en stack)
SOUS_CATS_STACK <- c("Monobank", "PrivatBank", "BuyMeACoffee", "WhitePay",
                     "Google Forms", "PayPal", "Oschadbank", "IBAN", "USDT (crypto)")

df_sous_stack <- df_sous_mois %>%
  filter(sous_cat %in% SOUS_CATS_STACK) %>%
  mutate(sous_cat = factor(as.character(sous_cat), levels = SOUS_CATS_STACK))

p_collecte_detail_stack <- ggplot(df_sous_stack, aes(x = mois, y = n_msg, fill = sous_cat)) +
  geom_col(width = 27, position = "fill") +
  scale_x_mois(breaks = "3 months") +
  scale_fill_manual(values = PAL_SOUS[SOUS_CATS_STACK], name = NULL) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  geom_phase_lines() +
  labs(
    title    = "Collecte — composition des mentions par sous-plateforme",
    subtitle = paste0(
      "Part de chaque plateforme dans les mentions collecte du mois (parmi les mois avec au moins une mention).\n",
      "Mêmes sources que Fig. 7a. Dignitas et Marikatefund exclus (1 message chacun)."
    ),
    x = NULL, y = "Part du mix collecte"
  )

save_plot(p_collecte_detail_stack,
          file.path(OUT, "17_crossposts_collecte_detail_stack.png"),
          format = "wide_16_9", width = 12, dpi = 300)

message("\n=== Terminé : 13 exports dans ", OUT, " ===")
