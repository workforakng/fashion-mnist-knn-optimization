#!/usr/bin/env bash
# Build the four per-member PDFs from the single source docs/paper.tex.
# Usage (from repo root):  bash docs/build_papers.sh
# Each member copy differs ONLY in the \StudentName / \StudentID / \StudentEmail
# macros. Members without a known email get an empty email line (no blank rule).
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# id|name|email|member-file-suffix|pdf-name
MEMBERS=(
  "241001001213|Akshar Nath Gorain|akshar241001001213@technoindiaeducation.com||AksharNathGorain"
  "241001001246|Sourodyuti Biswas Sanyal||_member2|SourodyutiBiswasSanyal"
  "241001001182|Aditi Barman||_member3|AditiBarman"
  "241001001274|Sanjana Mahata||_member4|SanjanaMahata"
)

for m in "${MEMBERS[@]}"; do
  IFS='|' read -r id name email suffix pdfname <<<"$m"
  tex="docs/paper${suffix}.tex"
  if [[ -n "$suffix" ]]; then
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
  fi
  job="paper${suffix}"
  for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" \
      -jobname="$job" "$tex" >/dev/null || { tail -40 "$BUILD/$job.log"; exit 1; }
  done
  cp "$BUILD/$job.pdf" "docs/${id}_${pdfname}.pdf"
  echo "built docs/${id}_${pdfname}.pdf ($(pdfinfo "docs/${id}_${pdfname}.pdf" | awk '/Pages/{print $2}') pages)"
done
