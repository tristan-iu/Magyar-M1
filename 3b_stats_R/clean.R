# install.packages("jsonlite") # si besoin
library(jsonlite)

# ── Configuration ─────────────────────────────────────────────────────────────
# INFILE : corpus enrichi canonique — chemin local à adapter (corpus disponible sur demande)
INFILE  <- "/chemin/vers/corpus/messages_clean.jsonl"

# OUTFILE : corpus allégé, écrit à côté de ce script — aucun réglage requis
this_file <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  if (length(f) > 0 && nzchar(f)) return(normalizePath(f, mustWork = FALSE))
  for (env in rev(sys.frames()))
    if (!is.null(env$ofile)) return(normalizePath(env$ofile, mustWork = FALSE))
  stop("Lancer via Rscript ou source().")
})
OUTFILE <- file.path(dirname(this_file), "messages_stripped.jsonl")
# ──────────────────────────────────────────────────────────────────────────────

infile  <- INFILE
outfile <- OUTFILE

# Champs à supprimer — texte lourd, chemin média, métadonnées non-analytiques.
drop_fields <- c(
  # Texte lourd (déjà disponible dans les fiches / SRT)
  "dialogue",                # transcription Whisper aplatie
  "dialogue_fr",             # traduction française du dialogue
  "ocr_texte",               # OCR complet sur keyframes
  "ocr_filigrane_texte",     # texte de watermark
  "reactions_detail",        # liste détaillée emoji×count
  # Chemin média (inutile hors NVME)
  "media_chemin",            # chemin vers le fichier média
  # Métadonnées non-analytiques
  "canal",                   # toujours "robert_magyar"
  "est_transfere",           # rarement utilisé
  "album_rang"               # rang dans l'album, pas exploité côté analyse
)

# Optionnel : sortie compressée
# outfile <- "messages_computervision_stripped.jsonl.gz"

con_in  <- file(infile, open = "r", encoding = "UTF-8")
con_out <- if (grepl("\\.gz$", outfile)) gzfile(outfile, open = "wt", encoding = "UTF-8") else
  file(outfile, open = "w", encoding = "UTF-8")

on.exit({
  try(close(con_in), silent = TRUE)
  try(close(con_out), silent = TRUE)
}, add = TRUE)

chunk_size <- 2000L
n_ok <- 0L
n_err <- 0L

repeat {
  lines <- readLines(con_in, n = chunk_size, warn = FALSE)
  if (length(lines) == 0L) break
  
  out_lines <- character(length(lines))
  
  for (i in seq_along(lines)) {
    li <- lines[[i]]
    if (!nzchar(li)) { out_lines[[i]] <- "" ; next }
    
    obj <- tryCatch(fromJSON(li, simplifyVector = FALSE), error = function(e) e)
    if (inherits(obj, "error")) {
      n_err <- n_err + 1L
      out_lines[[i]] <- li  # fallback: on réécrit la ligne inchangée
      next
    }
    
    # Suppression des champs lourds si présents
    for (f in drop_fields) if (!is.null(obj[[f]])) obj[[f]] <- NULL
    
    out_lines[[i]] <- toJSON(obj, auto_unbox = TRUE, null = "null")
    n_ok <- n_ok + 1L
  }
  
  writeLines(out_lines, con_out, sep = "\n")
}

message("Terminé. Lignes OK = ", n_ok, " | erreurs parse = ", n_err)
message("Fichier écrit : ", normalizePath(outfile, winslash = "/", mustWork = FALSE))
