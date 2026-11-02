#!/usr/bin/env bash
# Download raw datasets into data/raw. Safe to re-run (skips what exists).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw

# ---------- LIAR (Wang, 2017): 12,836 statements ----------
if [ ! -f liar/train.tsv ]; then
  mkdir -p liar
  if wget -q -O liar_dataset.zip https://www.cs.ucsb.edu/~william/data/liar_dataset.zip; then
    unzip -o -q liar_dataset.zip -d liar && rm liar_dataset.zip
    echo "[download] LIAR ok (UCSB zip)"
  else
    echo "[download] UCSB link failed. Get liar_dataset.zip from Kaggle ('LIAR dataset') and unzip into data/raw/liar/"
  fi
fi

# ---------- FakeNewsNet (Shu et al.): CSVs with news_url + title ----------
if [ ! -d FakeNewsNet ]; then
  git clone --depth 1 https://github.com/KaiDMML/FakeNewsNet.git
  echo "[download] FakeNewsNet CSVs ok"
fi

# ---------- TSHP-17 (Rashkin et al., 2017): source for Satire + Propaganda ----------
if [ ! -f tshp17/fulltrain.csv ]; then
  mkdir -p tshp17
  cat <<'EOF'
[download] TSHP-17 must be fetched manually (no stable direct link):
           1. Open https://homes.cs.washington.edu/~hrashkin/factcheck.html
           2. Download the news-article data (fulltrain.csv, balancedtest.csv)
           3. Put both files in data/raw/tshp17/
           Without it, Satire and Propaganda have no training data.
EOF
fi

# ---------- ISOT (sanity check set, used later) ----------
if [ ! -d isot ]; then
  echo "[download] ISOT (optional, week 4): get Fake.csv + True.csv from the University of Victoria ISOT page into data/raw/isot/"
fi
