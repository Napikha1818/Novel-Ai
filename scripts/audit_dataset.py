"""
audit_dataset.py
================
Audit kualitas dataset training.
Cek: teks rusak, kata tanpa spasi, artefak PDF, encoding error, dll.
"""

import json
import re
from pathlib import Path
from collections import Counter

TRAIN_PATH = Path("data/dataset/train.jsonl")


def load_data():
    data = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_broken_words(text):
    """Cek kata-kata yang terlalu panjang (kemungkinan tanpa spasi)."""
    words = text.split()
    broken = [w for w in words if len(w) > 35 and not w.startswith("http")]
    return broken


def check_encoding_artifacts(text):
    """Cek karakter encoding yang rusak."""
    artifacts = re.findall(r"[ï¿½\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", text)
    return artifacts


def check_pdf_artifacts(text):
    """Cek artefak khas PDF: nomor halaman nyasar, header/footer."""
    issues = []
    # Tanda hubung terpotong
    hyphen_breaks = re.findall(r"\w+­\w+", text)  # soft hyphen
    if hyphen_breaks:
        issues.extend(hyphen_breaks[:3])
    # Karakter aneh
    weird = re.findall(r"[©®™§¶]", text)
    if weird:
        issues.extend(weird)
    return issues


def check_empty_or_short(text, min_words=20):
    """Cek output yang terlalu pendek."""
    return len(text.split()) < min_words


def main():
    data = load_data()
    print(f"📊 Total pairs: {len(data)}\n")

    issues = {
        "broken_words": 0,
        "encoding_errors": 0,
        "pdf_artifacts": 0,
        "too_short": 0,
        "empty_output": 0,
    }

    broken_examples = []
    encoding_examples = []
    artifact_examples = []
    short_examples = []

    for i, pair in enumerate(data):
        output = pair["output"]
        instruction = pair["instruction"]
        meta = pair["metadata"]

        # Cek output kosong
        if not output.strip():
            issues["empty_output"] += 1
            continue

        # Cek terlalu pendek
        if check_empty_or_short(output):
            issues["too_short"] += 1
            short_examples.append((i, meta["type"], len(output.split()), output[:100]))

        # Cek broken words
        broken = check_broken_words(output)
        if broken:
            issues["broken_words"] += 1
            if len(broken_examples) < 5:
                broken_examples.append((i, meta["type"], broken[:3]))

        # Cek encoding
        enc_issues = check_encoding_artifacts(output)
        if enc_issues:
            issues["encoding_errors"] += 1
            if len(encoding_examples) < 5:
                encoding_examples.append((i, meta["type"], enc_issues[:3]))

        # Cek PDF artifacts
        pdf_issues = check_pdf_artifacts(output)
        if pdf_issues:
            issues["pdf_artifacts"] += 1
            if len(artifact_examples) < 5:
                artifact_examples.append((i, meta["type"], pdf_issues[:3]))

    # Laporan
    print("=" * 60)
    print("LAPORAN AUDIT DATASET")
    print("=" * 60)

    total_issues = sum(issues.values())
    clean = len(data) - total_issues
    print(f"\n✅ Bersih       : {clean}/{len(data)} ({clean/len(data)*100:.1f}%)")
    print(f"⚠️  Ada masalah  : {total_issues}/{len(data)} ({total_issues/len(data)*100:.1f}%)")

    print(f"\nDetail masalah:")
    for issue, count in issues.items():
        emoji = "🔴" if count > 50 else "🟡" if count > 10 else "🟢"
        print(f"  {emoji} {issue:20s}: {count}")

    if broken_examples:
        print(f"\n📋 Contoh broken words (kata tanpa spasi):")
        for idx, typ, words in broken_examples:
            print(f"  Pair #{idx} ({typ}): {words}")

    if encoding_examples:
        print(f"\n📋 Contoh encoding errors:")
        for idx, typ, chars in encoding_examples:
            print(f"  Pair #{idx} ({typ}): {chars}")

    if artifact_examples:
        print(f"\n📋 Contoh PDF artifacts:")
        for idx, typ, arts in artifact_examples:
            print(f"  Pair #{idx} ({typ}): {arts}")

    if short_examples:
        print(f"\n📋 Contoh output terlalu pendek (<20 kata):")
        for idx, typ, wc, preview in short_examples[:5]:
            print(f"  Pair #{idx} ({typ}, {wc} kata): {preview}...")

    # Cek sample teks untuk kualitas visual
    print(f"\n{'=' * 60}")
    print("SAMPLE TEKS UNTUK REVIEW MANUAL")
    print("=" * 60)

    for typ in ["writing", "explicit", "sliding_continuation", "style"]:
        for d in data:
            if d["metadata"]["type"] == typ:
                print(f"\n--- {typ.upper()} ---")
                print(f"Instruksi: {d['instruction'][:120]}...")
                print(f"Output   : {d['output'][:250]}...")
                break


if __name__ == "__main__":
    main()
