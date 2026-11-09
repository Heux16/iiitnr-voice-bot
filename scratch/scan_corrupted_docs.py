"""
scan_corrupted_docs.py — Scan every document in data/ for corruption, mojibake,
OCR artifacts, or 'needs_review' status flags.
"""

import os
import re
from pathlib import Path

DATA_DIR = Path(r"d:\iiit nr\sem 5\New folder (2)\uajy-academic-rag-chatbot\data")

corrupted_files = []
clean_files = []

# Mojibake patterns: Devanagari UTF-8 bytes read as Windows-1252 / ISO-8859-1: e.g. 'à¤', 'à¥', 'Ã©'
MOJIBAKE_REGEX = re.compile(r"à¤[^\s]{1,4}|à¥[^\s]{1,4}|Ã[¡-¿]|â€|â€˜|â€™|ï¬|ï¬‚")

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.endswith(".md"):
            continue
        filepath = Path(root) / f
        rel_path = filepath.relative_to(DATA_DIR)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as e:
            corrupted_files.append({
                "path": str(rel_path),
                "reason": f"Read error: {e}",
                "severity": "CRITICAL"
            })
            continue

        issues = []

        # Check frontmatter status
        if 'status: "needs_review"' in content or "status: needs_review" in content:
            issues.append("Frontmatter marked 'needs_review'")

        # Check for mojibake
        mojibake_matches = MOJIBAKE_REGEX.findall(content)
        if len(mojibake_matches) > 10:
            issues.append(f"Mojibake / encoding corruption detected ({len(mojibake_matches)} occurrences)")

        # Check for Tesseract gibberish (high concentration of random ASCII symbols)
        tesseract_gibberish = re.findall(r"(?:[a-zA-Z0-9]{1,2}[|«~@#$%^&*()_+=\[\]{}]{2,}[a-zA-Z0-9]{1,2})", content)
        if len(tesseract_gibberish) > 20:
            issues.append(f"Heavy OCR noise / scrambled symbols ({len(tesseract_gibberish)} instances)")

        # Check empty or tiny files
        if len(content.strip()) < 100:
            issues.append("File is practically empty (<100 characters)")

        if issues:
            corrupted_files.append({
                "path": str(rel_path),
                "issues": issues,
                "size_bytes": len(content),
                "mojibake_count": len(mojibake_matches)
            })
        else:
            clean_files.append(str(rel_path))

print(f"Scanned {len(clean_files) + len(corrupted_files)} total markdown documents.")
print(f"Clean documents: {len(clean_files)}")
print(f"Corrupted / Issue documents: {len(corrupted_files)}")
print("=" * 70)
for item in corrupted_files:
    print(f"FILE: {item['path']} ({item['size_bytes']} bytes)")
    for issue in item['issues']:
        print(f"  - {issue}")
    print("-" * 50)

