import json
from pathlib import Path

INPUT_DIR = Path("datasets/obrien")
OUTPUT_FILE = INPUT_DIR / "obrien_combined.json"

def main():
    if not INPUT_DIR.is_dir():
        print(f"Error: directory not found: {INPUT_DIR}")
        return

    # Use rglob to search recursively
    json_files = sorted(INPUT_DIR.rglob("*.json"))
    if not json_files:
        print(f"No .json files found in {INPUT_DIR}")
        return

    all_items = []
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            item = json.load(f)

        # Normalize key: target1 -> target_sentence_1
        if "target1" in item:
            item["target_sentence_1"] = item.pop("target1")

        # Remove unused fields
        for field in ("filler", "closing", "question", "target2"):
            item.pop(field, None)

        all_items.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"Combined {len(all_items)} items into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()