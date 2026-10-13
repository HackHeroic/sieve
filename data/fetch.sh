#!/bin/bash
BASE="https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
for Y in 2019 2020 2021 2022 2023 2024 2025 2026; do
  for H in Jan_Jun Jul_Dec; do
    F="EIA930_BALANCE_${Y}_${H}.csv"
    [ -s "$F" ] && { echo "skip $F"; continue; }
    if curl -sf -o "$F.tmp" "$BASE/$F"; then
      mv "$F.tmp" "$F"; echo "got $F $(du -h $F | cut -f1)"
    else
      rm -f "$F.tmp"; echo "miss $F"
    fi
  done
done
echo "DONE total: $(du -sh . | cut -f1)"
