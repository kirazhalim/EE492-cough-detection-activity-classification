from pathlib import Path
import pandas as pd
import shutil
import re

# =========================
# USER SETTINGS
# =========================
RAW_ROOT = Path(r"C:\Users\kiraz\OneDrive\Desktop\BOUN\Semester8(2025-2026)\EE492\data\raw_csv")
CURATED_ROOT = RAW_ROOT.parent / "curated_csv"
METADATA_PATH = RAW_ROOT.parent / "metadata.csv"

# Force subject and set defaults
SUBJECT = "kivanc"
DEFAULT_CONTEXT = "clean"
DEFAULT_CLOTHING = "overclothes"

# Activity map
ACTIVITY_MAP = {
    "sittingup": "sitting",
    "sitting": "sitting",
    "standingup": "standing",
    "standup": "standing",
    "standing": "standing",
    "walking": "walking",
    "running": "running",
}

# Context map
CONTEXT_MAP = {
    "coughnoise": "coughnoise",
    "musicnoise": "musicnoise",
    "sneezenoise": "sneezenoise",
    "snoozenoise": "snoozenoise",
    "doornoise": "doornoise",
    "clean": "clean",
    "unspecified": "clean", # Replaced unspecified with clean
}

def normalize_token(x: str) -> str:
    return x.strip().lower().replace(" ", "").replace("-", "")

def parse_old_filename(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Cannot parse filename: {filename}")

    date_raw = parts[0]
    local_record_id = parts[1]

    if not re.fullmatch(r"\d{8}", date_raw):
        raise ValueError(f"Invalid date format: {filename}")

    tail = parts[2:]
    tail_norm = [normalize_token(x) for x in tail]

    # Drop subject name if exists in filename
    if tail_norm and tail_norm[0] == SUBJECT:
        tail_norm.pop(0)

    if len(tail_norm) == 0:
        raise ValueError(f"No activity found: {filename}")

    activity_key = tail_norm.pop(0)
    if activity_key not in ACTIVITY_MAP:
        raise ValueError(f"Unknown activity '{activity_key}': {filename}")

    activity = ACTIVITY_MAP[activity_key]

    # Parse clothing and context
    clothing = DEFAULT_CLOTHING
    context_parts = []
    
    for token in tail_norm:
        if token == "underclothes":
            clothing = "underclothes"
        elif token == "overclothes":
            clothing = "overclothes"
        else:
            mapped_token = CONTEXT_MAP.get(token, token)
            context_parts.append(mapped_token)

    context = "_".join(context_parts) if context_parts else DEFAULT_CONTEXT
    date_iso = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

    return {
        "date_raw": date_raw,
        "date": date_iso,
        "local_record_id": int(local_record_id),
        "activity": activity,
        "context": context,
        "clothing": clothing,
    }

def build_new_filename(global_record_id: int, parsed: dict):
    # Keep filename short: omitted clothing, stored only in metadata
    rid = f"{global_record_id:03d}"
    return f"{rid}_{parsed['date_raw']}_{SUBJECT}_{parsed['activity']}_{parsed['context']}.csv"

def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Folder not found: {RAW_ROOT}")

    CURATED_ROOT.mkdir(parents=True, exist_ok=True)

    discovered = []
    errors = []

    # 1. Parse all files
    for csv_file in RAW_ROOT.rglob("*.csv"):
        try:
            parsed = parse_old_filename(csv_file.name)
            discovered.append({
                "source_path": csv_file,
                "source_filename": csv_file.name,
                **parsed
            })
        except Exception as e:
            errors.append({"source_file": str(csv_file), "error": str(e)})

    # 2. Sort chronologically
    discovered = sorted(
        discovered,
        key=lambda x: (x["date_raw"], x["local_record_id"], x["source_filename"])
    )

    metadata_rows = []

    # 3. Clear curated folder
    for f in CURATED_ROOT.glob("*.csv"):
        f.unlink()

    # 4. Copy files and build metadata
    for global_id, item in enumerate(discovered):
        new_filename = build_new_filename(global_id, item)
        dst = CURATED_ROOT / new_filename
        shutil.copy2(item["source_path"], dst)

        metadata_rows.append({
            "record_id": global_id,
            "filename": new_filename,
            "date": item["date"],
            "subject": SUBJECT,
            "activity": item["activity"],
            "context": item["context"],
            "clothing": item["clothing"],
            "relative_path": str(dst.relative_to(RAW_ROOT.parent)).replace("\\", "/"),
        })

    # 5. Save metadata
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_PATH, index=False)

    print(f"Curated files created: {len(df)}")
    print(f"Metadata saved to: {METADATA_PATH}")

    # 6. Save errors
    if errors:
        err_path = RAW_ROOT.parent / "prepare_dataset_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors saved to: {err_path}")

if __name__ == "__main__":
    main()