# Dépôt M1 SDHC
Dépôt git qui regroupe l'ensemble du code utilisé dans la collecte, enrichissement, et analyse du corpus. 
Corpus non inclu, disponible sur demande. 

# Structure du dépôt
---


```
├── 1a_scraper/          Collecte Telegram (Telethon) - JSONL brut + médias
├── 2a_audio/            Transcription Whisper large-v3 (GPU)
├── 2b_technique/        Métadonnées techniques ffprobe (résolution, durée, codec…)
├── 2c_vision/           Keyframes (ffmpeg), OCR cyrillique (EasyOCR), PySceneDetect
├── 2d_traduction/       Traduction SRT ukrainien français (DeepL / Mistral)
├── 3a_lexicometrie/     Lemmatisation spaCy (uk_core_news_trf) + TF-IDF par phase
└── 3b_quanti/           Analyses quantitatives et visualisations avec R
```

# Utilisation
---

Un README est inclu avec chaque script. Se référer au README inclue avec chaque protocole, mais la majorité fonctionne ainsi 

## Dépendances 
---
- Python 3.11+
- R 4.x avec les packages : `jsonlite`, `dplyr`, `ggplot2`, `lubridate`, `tidyr`, `scales`, `changepoint`, `bcp`
- ffmpeg / ffprobe 
- pytorch
- Modèle spaCy : `python -m spacy download uk_core_news_trf`
- Whisper
- EasyOCR
- Clé API DeepL pour la traduction 

Une carte graphique avec CUDA est recommandée pour l'utilisation de Whisper et EasyOCR. 

# Usage IA 
---
Les scripts de ce dépôt ont été développés avec l'assistance de Claude Code (Anthropic). Tous les scripts ont été relus et testés manuellement.
