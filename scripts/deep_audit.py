"""
deep_audit.py
=============
Audit mendalam:
1. Cek risiko memorisasi (model menghafal teks asli)
2. Cek keseimbangan dataset
3. Cek kualitas instruksi
4. Cek apakah alur novel writing sudah terdukung
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


def check_memorization_risk(data):
    """
    Cek risiko memorisasi: apakah output yang sama muncul berulang?
    Jika ya, model akan menghafal bukan belajar gaya.
    """
    print("=" * 60)
    print("1. RISIKO MEMORISASI")
    print("=" * 60)

    # Cek duplikasi output
    output_hashes = Counter()
    for d in data:
        # Hash dari 100 kata pertama output
        first_100 = " ".join(d["output"].split()[:100])
        output_hashes[first_100] += 1

    duplicates = {k: v for k, v in output_hashes.items() if v > 1}

    print(f"\n  Total pairs          : {len(data)}")
    print(f"  Output unik (100 kata): {len(output_hashes)}")
    print(f"  Output duplikat      : {len(duplicates)}")

    if duplicates:
        print(f"\n  ⚠️  Ada {len(duplicates)} output yang muncul lebih dari 1x:")
        for text, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"    {count}x: {text[:80]}...")
    else:
        print(f"\n  ✅ Tidak ada output duplikat persis")

    # Cek overlap antar tipe
    # Apakah teks yang sama dipakai di writing DAN explicit DAN style?
    output_by_type = {}
    for d in data:
        first_50 = " ".join(d["output"].split()[:50])
        typ = d["metadata"]["type"]
        if first_50 not in output_by_type:
            output_by_type[first_50] = set()
        output_by_type[first_50].add(typ)

    multi_type = {k: v for k, v in output_by_type.items() if len(v) > 1}
    print(f"\n  Output dipakai di >1 tipe: {len(multi_type)}")
    if multi_type:
        sample = list(multi_type.items())[:3]
        for text, types in sample:
            print(f"    Tipe {types}: {text[:60]}...")

    # Risiko: sliding window overlap
    sliding = [d for d in data if d["metadata"]["type"] == "sliding_continuation"]
    if sliding:
        # Cek berapa banyak overlap antar sliding pairs
        overlap_count = 0
        for i in range(len(sliding) - 1):
            words_a = set(sliding[i]["output"].split()[:50])
            words_b = set(sliding[i+1]["output"].split()[:50])
            overlap = len(words_a & words_b) / max(len(words_a), 1)
            if overlap > 0.5:
                overlap_count += 1

        print(f"\n  Sliding window pairs   : {len(sliding)}")
        print(f"  Pairs dengan >50% overlap: {overlap_count}")
        if overlap_count > len(sliding) * 0.3:
            print(f"  🟡 Overlap tinggi — stride mungkin terlalu kecil")
        else:
            print(f"  ✅ Overlap wajar")


def check_dataset_balance(data):
    """Cek keseimbangan dataset per tipe dan per novel."""
    print(f"\n{'=' * 60}")
    print("2. KESEIMBANGAN DATASET")
    print("=" * 60)

    # Per tipe
    type_counts = Counter(d["metadata"]["type"] for d in data)
    type_words = {}
    for d in data:
        t = d["metadata"]["type"]
        type_words[t] = type_words.get(t, 0) + len(d["output"].split())

    print(f"\n  Per tipe:")
    print(f"  {'Tipe':<25s} {'Pairs':>6s} {'%':>6s} {'Kata':>10s} {'Avg kata':>10s}")
    print(f"  {'-'*60}")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        words = type_words.get(t, 0)
        avg = words // count if count else 0
        emoji = "🟡" if pct > 50 else "🟢"
        print(f"  {emoji} {t:<23s} {count:>6d} {pct:>5.1f}% {words:>10,d} {avg:>10d}")

    # Per novel
    novel_counts = Counter(d["metadata"]["novel"] for d in data)
    print(f"\n  Per novel:")
    print(f"  {'Novel':<45s} {'Pairs':>6s} {'%':>6s}")
    print(f"  {'-'*60}")
    for novel, count in sorted(novel_counts.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {'🟢'} {novel:<43s} {count:>6d} {pct:>5.1f}%")


def check_instruction_quality(data):
    """Cek kualitas dan variasi instruksi."""
    print(f"\n{'=' * 60}")
    print("3. KUALITAS INSTRUKSI")
    print("=" * 60)

    # Cek variasi instruksi
    instruction_starts = Counter()
    for d in data:
        # Ambil 10 kata pertama instruksi
        start = " ".join(d["instruction"].split()[:10])
        instruction_starts[start] += 1

    print(f"\n  Total instruksi unik (10 kata pertama): {len(instruction_starts)}")
    print(f"\n  Top 10 instruksi paling sering:")
    for text, count in instruction_starts.most_common(10):
        print(f"    {count:>4d}x: {text[:70]}...")

    # Cek apakah instruksi mengandung teks novel (bocor)
    leak_count = 0
    for d in data:
        inst_words = set(d["instruction"].split())
        out_words = set(d["output"].split()[:30])
        overlap = len(inst_words & out_words)
        if overlap > 20:  # Terlalu banyak kata sama
            leak_count += 1

    print(f"\n  Instruksi yang 'bocor' (>20 kata sama dengan output): {leak_count}")
    if leak_count > len(data) * 0.1:
        print(f"  🔴 Terlalu banyak bocoran — model bisa menghafal")
    elif leak_count > 0:
        print(f"  🟡 Ada sedikit bocoran — perlu dicek")
    else:
        print(f"  ✅ Tidak ada bocoran")


def check_novel_workflow(data):
    """Cek apakah dataset mendukung alur penulisan novel."""
    print(f"\n{'=' * 60}")
    print("4. DUKUNGAN ALUR PENULISAN NOVEL")
    print("=" * 60)

    capabilities = {
        "Menulis dari instruksi": False,
        "Melanjutkan cerita": False,
        "Menulis adegan eksplisit": False,
        "Meniru gaya paragraf": False,
        "Konteks panjang (>1000 kata input)": False,
        "Output panjang (>500 kata)": False,
    }

    long_input = 0
    long_output = 0

    for d in data:
        t = d["metadata"]["type"]
        inst_words = len(d["instruction"].split())
        out_words = len(d["output"].split())

        if t == "writing":
            capabilities["Menulis dari instruksi"] = True
        if t in ("continuation", "sliding_continuation"):
            capabilities["Melanjutkan cerita"] = True
        if t in ("explicit", "explicit_aug"):
            capabilities["Menulis adegan eksplisit"] = True
        if t == "style":
            capabilities["Meniru gaya paragraf"] = True
        if inst_words > 1000:
            long_input += 1
            capabilities["Konteks panjang (>1000 kata input)"] = True
        if out_words > 500:
            long_output += 1
            capabilities["Output panjang (>500 kata)"] = True

    print(f"\n  Kemampuan yang didukung:")
    for cap, supported in capabilities.items():
        emoji = "✅" if supported else "❌"
        print(f"    {emoji} {cap}")

    print(f"\n  Pairs dengan input >1000 kata : {long_input}")
    print(f"  Pairs dengan output >500 kata : {long_output}")

    # Cek gap
    print(f"\n  ⚠️  GAP YANG TERDETEKSI:")
    gaps = []

    # Gap 1: Tidak ada instruksi "lanjutkan bab" dengan konteks ringkasan
    has_summary_context = False
    for d in data:
        if "ringkasan" in d["instruction"].lower() or "konteks" in d["instruction"].lower():
            has_summary_context = True
            break
    if not has_summary_context:
        gaps.append("Tidak ada pair dengan konteks ringkasan bab sebelumnya")

    # Gap 2: Tidak ada instruksi outline → bab
    has_outline = False
    for d in data:
        if "outline" in d["instruction"].lower() or "sinopsis" in d["instruction"].lower():
            has_outline = True
            break
    if not has_outline:
        gaps.append("Tidak ada pair outline/sinopsis → bab")

    # Gap 3: Tidak ada instruksi spesifik adegan
    has_scene_instruction = False
    for d in data:
        inst = d["instruction"].lower()
        if any(kw in inst for kw in ["adegan", "scene", "dialog", "percakapan"]):
            has_scene_instruction = True
            break
    if not has_scene_instruction:
        gaps.append("Tidak ada pair instruksi adegan spesifik")

    if gaps:
        for gap in gaps:
            print(f"    🔴 {gap}")
    else:
        print(f"    ✅ Tidak ada gap signifikan")


def main():
    data = load_data()
    check_memorization_risk(data)
    check_dataset_balance(data)
    check_instruction_quality(data)
    check_novel_workflow(data)

    print(f"\n{'=' * 60}")
    print("SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
