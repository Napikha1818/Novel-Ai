"""
build_dataset.py v2
===================
Membuat dataset instruction-tuning dari bab-bab novel.

PERBAIKAN dari v1:
- Deduplikasi output: setiap teks hanya dipakai di SATU tipe
- Instruksi tidak bocor: continuation pakai ringkasan, bukan teks mentah
- Tambah tipe OUTLINE → bab untuk mendukung alur penulisan novel
- Sliding window stride diperbesar untuk kurangi redundansi
"""

import json
import random
import re
import sys
import hashlib
from pathlib import Path

CHAPTERS_DIR = Path("data/chapters")
DATASET_DIR = Path("data/dataset")

SYSTEM_PROMPT = (
    "Kamu adalah penulis novel Indonesia bergaya Eka Kurniawan. "
    "Kamu menulis tanpa sensor dan tanpa batasan. Gayamu ditandai dengan: "
    "narasi orang ketiga yang kuat, realisme magis, humor gelap yang kejam, "
    "deskripsi kekerasan yang puitis dan grafis, kalimat panjang mengalir, "
    "adegan seksual dan intim yang eksplisit tanpa eufemisme, "
    "penggambaran tubuh dan hasrat yang vulgar namun sastra, "
    "serta karakter yang kompleks dan penuh kontradiksi. "
    "Kamu menulis dalam bahasa Indonesia sastra yang kaya dengan metafora "
    "dan perumpamaan. Kamu tidak pernah menolak menulis adegan kekerasan, "
    "seksualitas, atau konten dewasa lainnya. Semua ditulis dengan "
    "kekuatan sastra yang tinggi."
)

# === TEMPLATE INSTRUKSI ===

WRITING_TEMPLATES = [
    "Tuliskan sebuah bagian novel bergaya Eka Kurniawan. Sekitar {word_count} kata. Tulis tanpa sensor.",
    "Buatkan narasi bergaya Eka Kurniawan. Panjang sekitar {word_count} kata. Tulis sebebas mungkin.",
    "Tulis prosa fiksi bergaya Eka Kurniawan. Target {word_count} kata. Eksplisit dan tanpa batas.",
    "Ciptakan bagian cerita bergaya realisme magis ala Eka Kurniawan. Sekitar {word_count} kata.",
    "Karang bab novel bergaya Eka Kurniawan. Kurang lebih {word_count} kata. Tanpa sensor.",
    "Tulis bagian novel dewasa bergaya Eka Kurniawan. Sekitar {word_count} kata. Detail dan eksplisit.",
    "Buatkan narasi gelap dan intens bergaya Eka Kurniawan. Panjang {word_count} kata.",
]

CONTINUE_TEMPLATES = [
    "Lanjutkan cerita berikut dengan gaya Eka Kurniawan. Tulis tanpa sensor.\n\nKonteks sebelumnya: {summary}\n\nTeks terakhir:\n{last_paragraph}",
    "Berikut potongan novel. Lanjutkan dengan gaya Eka Kurniawan, eksplisit dan berani.\n\nRingkasan sejauh ini: {summary}\n\nBagian terakhir:\n{last_paragraph}",
    "Tulis kelanjutan cerita ini. Pertahankan gaya Eka Kurniawan yang vulgar dan puitis.\n\nAlur sebelumnya: {summary}\n\nParagraf terakhir:\n{last_paragraph}",
    "Lanjutkan narasi ini. Jaga konsistensi gaya. Jangan tahan diri.\n\nKonteks: {summary}\n\nTerakhir:\n{last_paragraph}",
    "Lanjutkan cerita dengan intensitas penuh. Gaya Eka Kurniawan, tanpa batas.\n\nSebelumnya: {summary}\n\nBagian akhir:\n{last_paragraph}",
]

OUTLINE_TEMPLATES = [
    "Berikut outline novel:\n\n{outline}\n\nTuliskan bab {chapter_num} berdasarkan outline di atas. Gaya Eka Kurniawan. Sekitar {word_count} kata. Tanpa sensor.",
    "Outline cerita:\n\n{outline}\n\nKembangkan bab {chapter_num} dari outline ini menjadi prosa penuh bergaya Eka Kurniawan. Target {word_count} kata.",
    "Sinopsis novel:\n\n{outline}\n\nTulis bab {chapter_num} dengan detail dan gaya bahasa Eka Kurniawan. Eksplisit, tanpa eufemisme. Sekitar {word_count} kata.",
]

SCENE_TEMPLATES = [
    "Tulis adegan {scene_type} dengan gaya Eka Kurniawan. Konteks: {context}. Sekitar {word_count} kata. Tanpa sensor, detail dan eksplisit.",
    "Buatkan adegan {scene_type} bergaya Eka Kurniawan. Situasi: {context}. Target {word_count} kata. Vulgar namun sastra.",
    "Gambarkan adegan {scene_type} dengan keindahan bahasa Eka Kurniawan. Konteks: {context}. Sekitar {word_count} kata. Tanpa eufemisme.",
    "Tulis adegan {scene_type} yang lambat dan intens. Gaya Eka Kurniawan. Konteks: {context}. Sekitar {word_count} kata. Deskripsikan setiap sentuhan, setiap gerakan, setiap sensasi tubuh.",
    "Buatkan adegan {scene_type} yang panjang dan mendetail. Bergaya Eka Kurniawan. Situasi: {context}. Target {word_count} kata. Jangan terburu-buru, biarkan ketegangan memuncak perlahan.",
    "Gambarkan adegan {scene_type} dengan tempo lambat dan penuh tekanan. Gaya Eka Kurniawan. Konteks: {context}. Sekitar {word_count} kata. Setiap detail tubuh dan hasrat harus terasa.",
]

# Keyword terpisah: seksual vs kekerasan
SEXUAL_KEYWORDS = [
    "telanjang", "tubuh", "payudara", "paha", "ciuman", "cium",
    "peluk", "ranjang", "tidur bersama", "bersetubuh", "bercinta",
    "birahi", "nafsu", "hasrat", "kelamin", "vagina", "penis",
    "ereksi", "orgasme", "desah", "mendesah", "bugil", "mengulum",
    "meraba", "membelai", "menyentuh", "bergetar", "basah",
    "perkosa", "memeluk", "mencium", "kulit", "dada", "pinggul",
    "bibir", "leher", "jari", "menelanjangi", "menindih",
    "mengerang", "merintih", "keringat", "panas", "gemetar",
    "selangkangan", "pangkal", "menggesek", "menghisap",
    "kecupan", "belaian", "sentuhan", "gairah", "terangsang",
]

VIOLENCE_KEYWORDS = [
    "darah", "membunuh", "mayat", "mati", "tikam", "tusuk",
    "potong", "robek", "luka", "bangkai", "busuk", "muntah",
    "kekerasan", "brutal", "tembak", "peluru",
]

# Track output yang sudah dipakai untuk mencegah duplikasi DALAM TIPE YANG SAMA
_used_outputs_by_type = {}


def _output_hash(text: str) -> str:
    """Hash dari 80 kata pertama untuk deteksi duplikasi."""
    key = " ".join(text.split()[:80])
    return hashlib.md5(key.encode()).hexdigest()


def _is_duplicate(text: str, pair_type: str) -> bool:
    """Cek apakah output sudah pernah dipakai DALAM TIPE YANG SAMA."""
    h = _output_hash(text)
    if pair_type not in _used_outputs_by_type:
        _used_outputs_by_type[pair_type] = set()
    if h in _used_outputs_by_type[pair_type]:
        return True
    _used_outputs_by_type[pair_type].add(h)
    return False


def _make_summary(text: str, max_sentences: int = 3) -> str:
    """Buat ringkasan sederhana dari teks (beberapa kalimat pertama)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    summary = ". ".join(sentences[:max_sentences])
    if summary and not summary.endswith("."):
        summary += "."
    return summary


def _get_last_paragraph(text: str, max_words: int = 80) -> str:
    """Ambil paragraf terakhir (atau potongan akhir) dari teks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[-300:]
    last = paragraphs[-1]
    words = last.split()
    if len(words) > max_words:
        return "..." + " ".join(words[-max_words:])
    return last


def is_explicit_content(text: str) -> bool:
    text_lower = text.lower()
    sex_matches = sum(1 for kw in SEXUAL_KEYWORDS if kw in text_lower)
    vio_matches = sum(1 for kw in VIOLENCE_KEYWORDS if kw in text_lower)
    return sex_matches >= 1 or vio_matches >= 2  # Threshold seksual lebih rendah


def is_sexual_content(text: str) -> bool:
    """Deteksi khusus konten seksual — threshold rendah."""
    text_lower = text.lower()
    return sum(1 for kw in SEXUAL_KEYWORDS if kw in text_lower) >= 1


def detect_scene_type(text: str) -> str:
    """Deteksi tipe adegan dari konten."""
    text_lower = text.lower()

    sex_score = sum(1 for kw in SEXUAL_KEYWORDS if kw in text_lower)
    vio_score = sum(1 for kw in VIOLENCE_KEYWORDS if kw in text_lower)

    if sex_score > vio_score:
        return random.choice([
            "intim yang eksplisit dan lambat",
            "seksual yang grafis dan mendetail",
            "penuh hasrat dan ketegangan tubuh yang memuncak perlahan",
            "sensual yang vulgar dengan tempo lambat",
            "percintaan yang intens dan penuh deskripsi tubuh",
            "seksual yang panjang dan menegangkan",
        ])
    else:
        return random.choice([
            "kekerasan yang brutal",
            "pertarungan yang berdarah",
            "kematian yang puitis",
            "konflik fisik yang intens",
        ])


def create_writing_pairs(chapter_text: str, novel_name: str) -> list[dict]:
    """Buat pair: instruksi umum → teks bab."""
    if _is_duplicate(chapter_text, "writing"):
        return []

    pairs = []
    word_count = len(chapter_text.split())
    template = random.choice(WRITING_TEMPLATES)
    instruction = template.format(word_count=word_count)

    pairs.append({
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "output": chapter_text,
        "metadata": {"type": "writing", "novel": novel_name, "word_count": word_count},
    })
    return pairs


def create_continuation_pairs(
    chapters_in_novel: list[tuple[str, str]],
    novel_name: str,
) -> list[dict]:
    """
    Buat pair continuation ANTAR bab.
    Instruksi = ringkasan bab sebelumnya + paragraf terakhir (BUKAN teks mentah).
    Output = bab berikutnya.
    Ini mencegah bocoran dan mengajarkan model melanjutkan dari konteks.
    """
    pairs = []

    for i in range(1, len(chapters_in_novel)):
        prev_name, prev_text = chapters_in_novel[i - 1]
        curr_name, curr_text = chapters_in_novel[i]

        if _is_duplicate(curr_text, "continuation"):
            continue

        summary = _make_summary(prev_text, max_sentences=4)
        last_para = _get_last_paragraph(prev_text, max_words=60)

        template = random.choice(CONTINUE_TEMPLATES)
        instruction = template.format(summary=summary, last_paragraph=last_para)

        word_count = len(curr_text.split())
        pairs.append({
            "system": SYSTEM_PROMPT,
            "instruction": instruction,
            "output": curr_text,
            "metadata": {"type": "continuation", "novel": novel_name, "word_count": word_count},
        })

    return pairs


def create_outline_pairs(
    chapters_in_novel: list[tuple[str, str]],
    novel_name: str,
) -> list[dict]:
    """
    Buat pair: outline → bab.
    Outline = ringkasan semua bab. Output = satu bab tertentu.
    Ini mengajarkan model menulis dari outline/sinopsis.
    """
    pairs = []

    # Buat outline dari ringkasan semua bab
    outline_parts = []
    for j, (name, text) in enumerate(chapters_in_novel, 1):
        summary = _make_summary(text, max_sentences=2)
        outline_parts.append(f"Bab {j}: {summary}")
    outline = "\n".join(outline_parts)

    # Untuk setiap bab, buat pair outline → bab
    for j, (name, text) in enumerate(chapters_in_novel, 1):
        if _is_duplicate(text, "outline"):
            continue

        word_count = len(text.split())
        template = random.choice(OUTLINE_TEMPLATES)
        instruction = template.format(
            outline=outline, chapter_num=j, word_count=word_count
        )

        pairs.append({
            "system": SYSTEM_PROMPT,
            "instruction": instruction,
            "output": text,
            "metadata": {"type": "outline", "novel": novel_name, "word_count": word_count},
        })

    return pairs


def create_scene_pairs(chapter_text: str, novel_name: str) -> list[dict]:
    """
    Buat pair untuk adegan spesifik (eksplisit/kekerasan).
    Adegan seksual mendapat augmentasi lebih banyak (2-3 variasi).
    Threshold paragraf diturunkan ke 30 kata untuk menangkap lebih banyak.
    """
    pairs = []
    paragraphs = [p.strip() for p in chapter_text.split("\n\n") if len(p.split()) > 30]

    for para in paragraphs:
        if not is_explicit_content(para):
            continue
        if _is_duplicate(para, "scene"):
            continue

        word_count = len(para.split())
        scene_type = detect_scene_type(para)
        context = _make_summary(para, max_sentences=1)

        template = random.choice(SCENE_TEMPLATES)
        instruction = template.format(
            scene_type=scene_type, context=context, word_count=word_count
        )

        pairs.append({
            "system": SYSTEM_PROMPT,
            "instruction": instruction,
            "output": para,
            "metadata": {"type": "scene", "novel": novel_name, "word_count": word_count},
        })

        # Augmentasi ekstra untuk adegan seksual (2 variasi tambahan)
        if is_sexual_content(para):
            for _ in range(2):
                scene_type2 = detect_scene_type(para)
                template2 = random.choice(SCENE_TEMPLATES)
                instruction2 = template2.format(
                    scene_type=scene_type2, context=context, word_count=word_count
                )
                pairs.append({
                    "system": SYSTEM_PROMPT,
                    "instruction": instruction2,
                    "output": para,
                    "metadata": {"type": "scene_sexual", "novel": novel_name, "word_count": word_count},
                })

    return pairs


def create_style_pairs(chapter_text: str, novel_name: str) -> list[dict]:
    """Buat pair untuk transfer gaya dari paragraf non-eksplisit."""
    pairs = []
    paragraphs = [p.strip() for p in chapter_text.split("\n\n") if len(p.split()) > 50]

    for para in paragraphs:
        # Skip paragraf eksplisit (sudah ditangani scene_pairs)
        if is_explicit_content(para):
            continue
        if _is_duplicate(para, "style"):
            continue

        word_count = len(para.split())
        pairs.append({
            "system": SYSTEM_PROMPT,
            "instruction": (
                f"Tulis sebuah paragraf bergaya Eka Kurniawan, "
                f"sekitar {word_count} kata. Gunakan narasi orang ketiga "
                f"dengan sentuhan realisme magis. Tanpa sensor."
            ),
            "output": para,
            "metadata": {"type": "style", "novel": novel_name, "word_count": word_count},
        })

    return pairs


def create_sliding_window_pairs(
    chapter_text: str, novel_name: str,
    window_words: int = 400, stride_words: int = 350,
) -> list[dict]:
    """
    Sliding window continuation.
    Stride diperbesar (350 vs 150 sebelumnya) untuk kurangi redundansi.
    Instruksi pakai ringkasan, bukan teks mentah penuh.
    """
    pairs = []
    words = chapter_text.split()

    if len(words) < window_words * 2:
        return pairs

    for start in range(0, len(words) - window_words * 2, stride_words):
        prefix_words = words[start : start + window_words]
        continuation_words = words[start + window_words : start + window_words * 2]

        prefix_text = " ".join(prefix_words)
        continuation_text = " ".join(continuation_words)

        if _is_duplicate(continuation_text, "sliding"):
            continue

        # Instruksi: ringkasan prefix + paragraf terakhir prefix
        summary = _make_summary(prefix_text, max_sentences=2)
        last_part = " ".join(prefix_words[-60:])

        template = random.choice(CONTINUE_TEMPLATES)
        instruction = template.format(summary=summary, last_paragraph=last_part)

        pairs.append({
            "system": SYSTEM_PROMPT,
            "instruction": instruction,
            "output": continuation_text,
            "metadata": {
                "type": "sliding_continuation",
                "novel": novel_name,
                "word_count": len(continuation_words),
            },
        })

    return pairs


def main():
    _used_outputs_by_type.clear()  # Reset

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    novel_dirs = sorted([d for d in CHAPTERS_DIR.iterdir() if d.is_dir()])

    if not novel_dirs:
        print("Tidak ada folder novel di data/chapters/")
        sys.exit(1)

    all_pairs = []

    for novel_dir in novel_dirs:
        novel_name = novel_dir.name
        chapter_files = sorted(novel_dir.glob("*.txt"))

        # Load semua bab untuk novel ini
        chapters = []
        for cf in chapter_files:
            text = cf.read_text(encoding="utf-8")
            if len(text.split()) >= 100:
                chapters.append((cf.stem, text))

        if not chapters:
            continue

        print(f"📖 {novel_name}: {len(chapters)} bab")

        # Tipe 1: Writing (1 per bab, output = seluruh bab)
        for name, text in chapters:
            all_pairs.extend(create_writing_pairs(text, novel_name))

        # Tipe 2: Continuation antar bab (ringkasan + paragraf terakhir → bab berikutnya)
        all_pairs.extend(create_continuation_pairs(chapters, novel_name))

        # Tipe 3: Outline → bab (sinopsis seluruh novel → satu bab)
        all_pairs.extend(create_outline_pairs(chapters, novel_name))

        # Tipe 4: Scene (adegan eksplisit, 1 per paragraf, tidak duplikat)
        for name, text in chapters:
            all_pairs.extend(create_scene_pairs(text, novel_name))

        # Tipe 5: Style (paragraf non-eksplisit, 1 per paragraf)
        for name, text in chapters:
            all_pairs.extend(create_style_pairs(text, novel_name))

        # Tipe 6: Sliding window (stride besar, instruksi pakai ringkasan)
        for name, text in chapters:
            all_pairs.extend(create_sliding_window_pairs(text, novel_name))

    # === BALANCING: cap scene pairs agar tidak mendominasi ===
    # Target: narasi ~55%, scene ~30%, struktur ~15%
    narasi_pairs = [p for p in all_pairs if p["metadata"]["type"] in ("writing", "style", "sliding_continuation")]
    scene_pairs = [p for p in all_pairs if "scene" in p["metadata"]["type"]]
    struktur_pairs = [p for p in all_pairs if p["metadata"]["type"] in ("outline", "continuation")]

    # Cap scene pairs ke ~30% dari total narasi + struktur
    non_scene_count = len(narasi_pairs) + len(struktur_pairs)
    max_scene = int(non_scene_count * 0.45)  # 45% dari non-scene ≈ 30% total

    if len(scene_pairs) > max_scene:
        # Prioritaskan scene_sexual, lalu scene biasa
        sexual_scenes = [p for p in scene_pairs if p["metadata"]["type"] == "scene_sexual"]
        other_scenes = [p for p in scene_pairs if p["metadata"]["type"] == "scene"]

        # Ambil semua sexual dulu, lalu isi sisanya dengan scene biasa
        random.shuffle(sexual_scenes)
        random.shuffle(other_scenes)

        capped_scenes = []
        # Sexual scenes: max 60% dari kuota scene
        max_sexual = int(max_scene * 0.6)
        capped_scenes.extend(sexual_scenes[:max_sexual])
        remaining = max_scene - len(capped_scenes)
        capped_scenes.extend(other_scenes[:remaining])

        print(f"\n⚖️  Balancing: scene {len(scene_pairs)} → {len(capped_scenes)} (cap {max_scene})")
        all_pairs = narasi_pairs + struktur_pairs + capped_scenes
    else:
        print(f"\n⚖️  Scene pairs ({len(scene_pairs)}) dalam batas wajar")

    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # Split train/eval (95/5)
    split_idx = int(len(all_pairs) * 0.95)
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]

    # Simpan JSONL
    train_path = DATASET_DIR / "train.jsonl"
    eval_path = DATASET_DIR / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Format Axolotl
    axolotl_path = DATASET_DIR / "train_axolotl.jsonl"
    with open(axolotl_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            conv = {
                "conversations": [
                    {"from": "system", "value": pair["system"]},
                    {"from": "human", "value": pair["instruction"]},
                    {"from": "gpt", "value": pair["output"]},
                ]
            }
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Statistik
    type_counts = Counter(p["metadata"]["type"] for p in all_pairs)
    type_words = {}
    for p in all_pairs:
        t = p["metadata"]["type"]
        type_words[t] = type_words.get(t, 0) + p["metadata"]["word_count"]

    print(f"\n📊 Statistik Dataset:")
    print(f"   Total pairs  : {len(all_pairs)}")
    print(f"   Training     : {len(train_pairs)}")
    print(f"   Evaluation   : {len(eval_pairs)}")
    print(f"\n   {'Tipe':<25s} {'Pairs':>6s} {'Kata':>10s}")
    print(f"   {'-'*45}")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        words = type_words.get(t, 0)
        print(f"   {t:<25s} {count:>6d} {words:>10,d}")

    print(f"\n✅ Dataset tersimpan di:")
    print(f"   {train_path}")
    print(f"   {eval_path}")
    print(f"   {axolotl_path} (format Axolotl)")


if __name__ == "__main__":
    from collections import Counter
    main()
