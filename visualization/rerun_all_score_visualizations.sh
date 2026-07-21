#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
BOOK="${1:-alice_wonderland}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python virtual environment not found: $PYTHON" >&2
  exit 1
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurrative-matplotlib-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/neurrative-numba-cache}"

mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT_ROOT"

echo "Regenerating score visualizations for: $BOOK"

"$PYTHON" visualization/scores.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_field.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_field_3d.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_field_umap_2d.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_field_umap_3d.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_fields_pca_by_chapter.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_fields_pca_chapter_comparison.py --book "$BOOK"
"$PYTHON" visualization/scores_vector_fields_pca_chapter_comparison_3d.py --book "$BOOK"

echo "Done. Outputs are in: $PROJECT_ROOT/output/$BOOK/scores"
