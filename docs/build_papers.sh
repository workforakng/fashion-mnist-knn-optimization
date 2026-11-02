#!/usr/bin/env bash
# Build the four per-member PDFs from the single source docs/paper.tex.
# Usage (from repo root):  bash docs/build_papers.sh
# Each member copy differs ONLY in the \StudentName / \StudentID / \StudentEmail
# macros. Intermediate files are kept strictly in a temporary build directory.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# id|name|email|pdf-name
MEMBERS=(
  "241001001213|Akshar Nath Gorain|akshar241001001213@technoindiaeducation.com|AksharNathGorain"
  "241001001246|Sourodyuti Biswas Sanyal||SourodyutiBiswasSanyal"
  "241001001182|Aditi Barman||AditiBarman"
  "241001001274|Sanjana Mahata||SanjanaMahata"
)

mkdir -p pdf

for m in "${MEMBERS[@]}"; do
  IFS='|' read -r id name email pdfname <<<"$m"
  tex="$BUILD/paper_${id}.tex"
  python3 - "$id" "$name" "$email" "$tex" <<'PY'
import re, sys
sid, name, email, out = sys.argv[1:]
src = open("docs/paper.tex", encoding="utf-8").read()
def sub(macro, val, s):
    pat = re.compile(r"\\newcommand\{\\%s\}\{.*\}" % macro)
    assert pat.search(s), macro
    return pat.sub(lambda _: "\\newcommand{\\%s}{%s}" % (macro, val), s, count=1)
src = sub("StudentName", name, src)
src = sub("StudentID", sid, src)
src = sub("StudentEmail", email, src)
open(out, "w", encoding="utf-8").write(src)
PY

  job="paper_${id}"
  for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" \
      -jobname="$job" "$tex" >/dev/null || { tail -40 "$BUILD/$job.log"; exit 1; }
  done
  
  # Copy to docs/ and to pdf/
  cp "$BUILD/$job.pdf" "docs/${id}_${pdfname}.pdf"
  cp "$BUILD/$job.pdf" "pdf/${id}_FashionKNN.pdf"
  echo "built ${id}_${pdfname}.pdf -> docs/ and pdf/ ($(pdfinfo "$BUILD/$job.pdf" | awk '/Pages/{print $2}') pages)"
done

# Keep matching Akshar submission copy in docs
cp "pdf/241001001213_FashionKNN.pdf" "docs/241001001213_FashionKNN.pdf"
cp "pdf/241001001213_FashionKNN.pdf" "241001001213_FashionKNN.pdf"
