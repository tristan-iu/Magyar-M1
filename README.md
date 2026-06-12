# Dépôt M1 SDHC 
Ensemble du code utilisé dans le cadre du mémoire de M1 SDHC. 

Le corpus porte sur les publications de la chaîne Telegram du commandant de drones ukrainien Robert « *Madyar* » Brovdi ([@robert_magyar](https://t.me/robert_magyar)), de septembre 2022 à septembre 2025 — 1365 messages couvrant trois phases de production (artisanale, semi-professionnelle, institutionnelle). Le corpus n'est pas inclus dans le dépôt ; la chaîne est publique et la collecte reproductible.

## Structure du dépôt

La numérotation correspond à l'ordre d'exécution : (1) collecte, (2) enrichissement, (3) analyse. Le scraper produit un JSONL de base ; les étapes suivantes enrichissent incrémentalement ce fichier avec de nouveaux champs.

```
├── 0_config/            Configuration centrale (config.yaml, utils.py)
├── 1a_scraper/          Collecte Telegram via l'API officielle et la bibliothèque Telethon 
├── 2a_metadonnees/      Métadonnées techniques via ffprobe (résolution, durée, bitrate…)
├── 2b_transcription/    Transcription via Whisper 
├── 2c_traduction/       Traduction des .srt de l'ukrainien au français (DeepL / LM Studio)
├── 2d_vision/           Keyframes, OCR cyrillique, SceneDetect, InsightFace, CLIP, blasons
│   ├── keyframes/             Extraction keyframes + détection de scènes
│   │   ├── keyframer.py           Pipeline keyframes fixes (ffmpeg, 1/10s) + OCR + SceneDetect
│   │   └── scene_detect.py        Détection de scènes standalone (HistogramDetector fallback)
│   ├── visages/               Détection des visages via InsightFace 
│   ├── clip/                  Classification zero-shot via CLIP
│   └── blasons/               Détection logos de brigade via SIFT & RANSAC
├── 3a_lexicometrie/     Analyse textuelle (lemmatisation spaCy, TF-IDF, spécificités, LDA, AFC, CAH)
├── 3b_stats_R/          Analyses quantitatives et figures du mémoire (R + ggplot2)
└── 3c_couleurs/         Analyse colorimétrique HSV des keyframes (entropie, PCA, similarité)
```

## Utilisation
Copier le template et adapter les chemins locaux (`paths.*`). `0_config/config.example.yaml` documente la structure complète.

Se référer au README de chaque module. Tous les scripts sont paramétrables via CLI, du type :
```bash
python3 script.py --input corpus.jsonl --output corpus_enrichi.jsonl
```

L'utilisation sur d'autres chaînes Telegram n'est pas garantie, en particulier pour les langues autres que l'ukrainien (modifier le prompt Whisper, les modèles spaCy et OCR). 

## Dépendances

```bash
pip install -r requirements.txt        # dépendances Python
python3 -m spacy download uk_core_news_trf  # modèle spaCy ukrainien 
sudo apt install ffmpeg                 # dépendance système
```

Un GPU CUDA est recommandé pour Whisper, EasyOCR et InsightFace.

Les modules qui nécessitent une clé API (scraper Telegram, traduction DeepL) lisent leurs credentials depuis un fichier `.env`. Un `.env.example` est fourni dans chaque module concerné.

## Usage IA

Les scripts ont été développés avec l'assistance de Claude Code (Anthropic, versions Sonnet 4.5-6/Opus 4.5-6, novembre 2025 à juin 2026) sur des tâches d'implémentation technique (écriture et debug de code Python et R, refactorisations). Chacun a été relu, commenté et testé manuellement ; les décisions méthodologiques (choix d'outils, seuils, périodisation) relèvent de l'auteur.

## Licence

Code sous licence [MIT](LICENSE). Le corpus (textes, médias, transcriptions) n'est pas couvert par cette licence et n'est pas redistribué.
