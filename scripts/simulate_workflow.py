"""
simulate_workflow.py
====================
Simulasi alur kerja penulisan novel.
Mengecek apakah dataset kita mendukung setiap langkah.

Alur yang diharapkan user:
1. User: "Ini outline novelku: ..."
2. User: "Tulis bab 1 tentang X, sekitar 2000 kata"
3. Model: [menulis bab 1]
4. User: "Lanjutkan ke bab 2, tambahkan adegan seks doggy style yang detail"
5. Model: [menulis bab 2 dengan konteks bab 1]
6. User: "Lanjutkan bab 3, karakter A bertemu B, ada konflik"
7. Model: [menulis bab 3 dengan konteks bab 1-2]
... dan seterusnya
"""

import json
from pathlib import Path
from collections import Counter

TRAIN_PATH = Path("data/dataset/train.jsonl")


def load_data():
    data = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def simulate(data):
    print("=" * 60)
    print("SIMULASI ALUR PENULISAN NOVEL")
    print("=" * 60)

    # === LANGKAH 1: Set outline ===
    print("\n📋 LANGKAH 1: User memberikan outline")
    print("   Contoh: 'Ini outline novelku: Bab 1 tentang X, Bab 2 tentang Y...'")
    outline_pairs = [d for d in data if d["metadata"]["type"] == "outline"]
    print(f"   Dataset mendukung? {'✅' if outline_pairs else '❌'} ({len(outline_pairs)} pairs)")
    if outline_pairs:
        sample = outline_pairs[0]
        inst_preview = sample["instruction"][:150]
        print(f"   Contoh instruksi: {inst_preview}...")

    # === LANGKAH 2: Tulis bab 1 ===
    print("\n📝 LANGKAH 2: 'Tulis bab 1 tentang X, sekitar 2000 kata'")
    writing_pairs = [d for d in data if d["metadata"]["type"] == "writing"]
    long_writing = [d for d in writing_pairs if d["metadata"]["word_count"] > 1500]
    print(f"   Dataset mendukung? {'✅' if writing_pairs else '❌'} ({len(writing_pairs)} pairs)")
    print(f"   Output >1500 kata: {len(long_writing)} pairs")
    if writing_pairs:
        avg_words = sum(d["metadata"]["word_count"] for d in writing_pairs) // len(writing_pairs)
        print(f"   Rata-rata output: {avg_words} kata")

    # === LANGKAH 3: Lanjutkan dengan konteks ===
    print("\n🔄 LANGKAH 3: 'Lanjutkan ke bab 2, tambahkan adegan seks detail'")
    cont_pairs = [d for d in data if d["metadata"]["type"] == "continuation"]
    print(f"   Continuation pairs: {'✅' if cont_pairs else '❌'} ({len(cont_pairs)} pairs)")

    # Cek apakah instruksi continuation mengandung ringkasan (bukan teks mentah)
    has_summary = 0
    has_raw_text = 0
    for d in cont_pairs:
        inst = d["instruction"]
        if "ringkasan" in inst.lower() or "konteks" in inst.lower() or "sebelumnya" in inst.lower():
            has_summary += 1
        if len(inst.split()) > 300:
            has_raw_text += 1

    print(f"   Pakai ringkasan: {has_summary} pairs")
    print(f"   Pakai teks mentah panjang (>300 kata): {has_raw_text} pairs")
    if has_summary > has_raw_text:
        print(f"   ✅ Mayoritas pakai ringkasan — model belajar dari konteks, bukan menghafal")
    else:
        print(f"   🟡 Banyak teks mentah — risiko memorisasi")

    # === LANGKAH 4: Sliding context untuk bab-bab selanjutnya ===
    print("\n📖 LANGKAH 4: Bab 3, 4, 5... (sliding context)")
    sliding = [d for d in data if d["metadata"]["type"] == "sliding_continuation"]
    print(f"   Sliding pairs: {'✅' if sliding else '❌'} ({len(sliding)} pairs)")

    # === LANGKAH 5: Adegan eksplisit on-demand ===
    print("\n🔥 LANGKAH 5: 'Tambahkan adegan seks doggy style yang detail'")
    scene_pairs = [d for d in data if d["metadata"]["type"] == "scene"]
    print(f"   Scene pairs: {'✅' if scene_pairs else '❌'} ({len(scene_pairs)} pairs)")

    # Cek variasi adegan
    scene_types = Counter()
    for d in scene_pairs:
        inst = d["instruction"].lower()
        if "intim" in inst or "seksual" in inst or "hasrat" in inst or "sensual" in inst:
            scene_types["seksual/intim"] += 1
        elif "kekerasan" in inst or "pertarungan" in inst or "kematian" in inst or "konflik" in inst:
            scene_types["kekerasan/aksi"] += 1
        else:
            scene_types["lainnya"] += 1

    print(f"   Variasi adegan:")
    for st, count in scene_types.most_common():
        print(f"     {st}: {count} pairs")

    # === LANGKAH 6: Gaya paragraf ===
    print("\n✍️  LANGKAH 6: Konsistensi gaya Eka Kurniawan")
    style_pairs = [d for d in data if d["metadata"]["type"] == "style"]
    print(f"   Style pairs: {'✅' if style_pairs else '❌'} ({len(style_pairs)} pairs)")

    # === RINGKASAN ===
    print(f"\n{'=' * 60}")
    print("RINGKASAN")
    print("=" * 60)

    steps = [
        ("Outline → bab", bool(outline_pairs)),
        ("Tulis bab dari instruksi", bool(writing_pairs)),
        ("Lanjutkan dengan konteks", bool(cont_pairs) and has_summary > 0),
        ("Sliding context bab panjang", bool(sliding)),
        ("Adegan eksplisit on-demand", bool(scene_pairs)),
        ("Konsistensi gaya", bool(style_pairs)),
    ]

    all_ok = True
    for step, ok in steps:
        emoji = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {emoji} {step}")

    if all_ok:
        print(f"\n  🎉 Semua langkah terdukung oleh dataset!")
    else:
        print(f"\n  ⚠️  Ada langkah yang belum terdukung")

    # === GAP ANALYSIS ===
    print(f"\n{'=' * 60}")
    print("GAP ANALYSIS — Apa yang masih kurang?")
    print("=" * 60)

    gaps = []

    # Gap 1: Tidak ada multi-turn conversation
    print(f"\n  🟡 Multi-turn conversation:")
    print(f"     Dataset kita hanya 1 turn (instruksi → output).")
    print(f"     Saat user bilang 'lanjutkan bab 2' setelah bab 1,")
    print(f"     generate_novel.py yang menangani context management,")
    print(f"     BUKAN model. Ini sudah benar — model hanya perlu")
    print(f"     menulis bagus dari instruksi + konteks yang diberikan.")

    # Gap 2: Instruksi spesifik adegan
    specific_scene = [d for d in data if "adegan" in d["instruction"].lower()]
    if len(specific_scene) < 50:
        print(f"\n  🟡 Instruksi adegan spesifik: hanya {len(specific_scene)} pairs")
        print(f"     User mungkin bilang: 'tulis adegan seks doggy style'")
        print(f"     Tapi dataset tidak punya instruksi se-spesifik itu.")
        print(f"     Model akan tetap bisa menulis karena base model paham,")
        print(f"     tapi mungkin kurang 'berani' di detail spesifik.")
        gaps.append("instruksi_adegan_spesifik")

    # Gap 3: Dialog
    dialog_count = sum(1 for d in data if '"' in d["output"][:200])
    print(f"\n  {'✅' if dialog_count > 100 else '🟡'} Dialog dalam output: {dialog_count} pairs")
    if dialog_count < 100:
        print(f"     Model mungkin kurang kuat menulis dialog.")
        gaps.append("dialog")

    return gaps


if __name__ == "__main__":
    data = load_data()
    gaps = simulate(data)
