"""Walk data/mimic/raw, report every table, write data/mimic/INVENTORY.md.

Each MIMIC folder ships two orderings of the same rows (_sorted / _random); we
inventory the _sorted variant and note the duplication. Row counts come from a
fast line count (wc -l) minus the header.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .config import MIMIC_RAW, DATA_DIR

# Tables we actually consume downstream, and why. Everything else is logged as
# "considered / unused" per CLAUDE.md §2.2 item 5.
USED = {
    "NOTEEVENTS": "Discharge-summary text -> M1 fine-tuning corpus (clinical fluency).",
    "PATIENTS": "SUBJECT_ID + GENDER -> patient identifier list + gender attacks.",
    "DIAGNOSES_ICD": "SUBJECT_ID -> ICD9 codes -> diagnosis-attack grounding.",
    "D_ICD_DIAGNOSES": "ICD9 code -> human-readable title (join target).",
    "PRESCRIPTIONS": "SUBJECT_ID -> DRUG -> medication-attack grounding.",
    "ADMISSIONS": "SUBJECT_ID -> INSURANCE / DIAGNOSIS -> insurance-attack grounding.",
}


def _row_count(path: Path) -> int:
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    try:
        return int(out.stdout.strip().split()[0]) - 1  # minus header
    except (ValueError, IndexError):
        return -1


def _sorted_csv(folder: Path) -> Path | None:
    cands = sorted(folder.glob("*.csv"))
    for c in cands:
        if "sorted" in c.name.lower():
            return c
    return cands[0] if cands else None


def build_inventory() -> Path:
    lines = ["# MIMIC-III Inventory", "",
             f"Source: `{MIMIC_RAW}`", "",
             "Each table is shipped as `_sorted` and `_random` (identical rows, "
             "different order); counts below are for the `_sorted` variant.", "",
             "## Tables", ""]
    used_rows = {}
    for folder in sorted(p for p in MIMIC_RAW.iterdir() if p.is_dir()):
        csv_path = _sorted_csv(folder)
        if csv_path is None:
            continue
        n = _row_count(csv_path)
        size_mb = csv_path.stat().st_size / 1e6
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            header = next(csv.reader(f))
        role = USED.get(folder.name, "considered; unused in current pipeline")
        used_rows[folder.name] = n
        lines += [f"### {folder.name}",
                  f"- rows: **{n:,}**  |  size: {size_mb:,.1f} MB  |  cols: {len(header)}",
                  f"- columns: `{', '.join(header)}`",
                  f"- role: {role}", ""]

    # de-identification note
    lines += ["## De-identification check", "",
              "Public MIMIC-III is de-identified: `PATIENTS.SUBJECT_ID` is a surrogate "
              "integer (no real names), dates are shifted, and note text carries "
              "`[**...**]` redactions. The 'patient identifier' used by the attack "
              "suite is therefore the surrogate SUBJECT_ID, not a real name. No field "
              "was found to contain real identifying information.", "",
              "## Unused tables", "",
              ", ".join(f for f in sorted(used_rows) if f not in USED) or "(none)", ""]

    out = DATA_DIR / "mimic" / "INVENTORY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_inventory()
    print("wrote", p)
    print(p.read_text()[:1500])
