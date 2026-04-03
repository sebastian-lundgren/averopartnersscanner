#!/usr/bin/env python3
"""
Fase 3 – plassholder for retrening på TrainingExample-rader.

Kjør etter at du har samlet nok rettelser i SQLite-tabellen training_examples.
Denne MVP-en lagrer kun datasettet; bytt ut med ekte treningsløkke (PyTorch/ultralytics osv.).
"""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "app.db"
OUT = ROOT / "data" / "export_training_manifest.jsonl"


def main():
    if not DB.is_file():
        print("Ingen database funnet:", DB)
        return
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM training_examples ORDER BY created_at DESC"
    ).fetchall()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), default=str) + "\n")
    print(f"Skrev {len(rows)} rader til {OUT}")


if __name__ == "__main__":
    main()
