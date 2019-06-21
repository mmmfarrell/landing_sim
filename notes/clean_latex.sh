#!/bin/bash

EXTENSIONS=".aux .log .nav .out .snm .synctex.gz .toc .bbl .blg -blx.bib .bcf .run.xml .acn .acr .alg .glg .glo .gls .ist .lof .lot .auxlock .synctex.gz(busy) .vrb .fls .fdb_latexmk .latexmk .ind .idx .ilg"

MAXDEPTH="-maxdepth 1"
if [ "$1" = "-r" ] || [ "$1" = "-R" ]; then
  MAXDEPTH=""
fi

find . $MAXDEPTH -type f -name "*.tex" | while read FILE
do

  DIRNAME=$(dirname "$FILE")
  FILENAME=$(basename "$FILE" .tex)

  for EXT in $EXTENSIONS; do
    rm -vf "$DIRNAME/$FILENAME$EXT"
  done

done
