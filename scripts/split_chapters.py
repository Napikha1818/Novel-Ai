"""
split_chapters.py
=================
Memotong teks hasil ekstraksi menjadi potongan-potongan yang optimal
untuk training LLM.

Strategi:
1. Coba deteksi bab dari pola teks (angka romawi, "Bab X", dll)
2. Jika tidak ada pola bab → potong di batas paragraf alami
   dengan target ~2000-3000 kata per potongan
3. Pastikan potongan tidak memotong di tengah kalimat

Hasil disimpan di data/chapters/<nama_novel>/bab_01.txt, dst.
"""

import re
import sys
from pathlib import Path

EXTRACTED_DIR = Path("data/extracted")
CHAPTERS_DIR = Path("data/chapters")

# Target ukuran per potongan (dalam kata)
TARGET_CHUNK_WORDS = 2500
MIN_CHUNK_WORDS = 800
MAX_CHUNK_WORDS = 5000

# Pola-pola penanda bab dalam novel Indonesia
CHAPTER_PATTERNS = [
    r"---BAB_SEPARATOR---",
    r"(?:^|\n)(?:BAB|Bab|bab)\s+[\dIVXLCDMivxlcdm]+[.\s:]*\n",
    r"(?:^|\n)(?:Bagian|BAGIAN)\s+(?:Pertama|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|Kesembilan|Kesepuluh|Satu|Dua|Tiga|Empat|Lima)\s*\n",
]


def try_chapter_split(text: str) -> list[str] | None:
    """Coba split berdasarkan pola bab. Return None jika gagal."""
    for pattern in CHAPTER_PATTERNS:
        splits = re.split(pattern, text)
        meaningful = [s.strip() for s in splits if len(s.strip().split()) > MIN_CHUNK_WORDS]
        if len(meaningful) >= 3:  # Minimal 3 bab baru dianggap berhasil
            return meaningful
    return None


def smart_split(text: str) -> list[str]:
    """
    Potong teks di batas paragraf alami dengan target ~2500 kata.
    Tidak pernah memotong di tengah paragraf.
    """
    # Split berdasarkan paragraf (double newline)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # Jika menambah paragraf ini melebihi MAX, simpan chunk saat ini
        if current_words + para_words > MAX_CHUNK_WORDS and current_words >= MIN_CHUNK_WORDS:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_words = para_words
        # Jika sudah mencapai target dan paragraf berikutnya cukup besar
        elif current_words >= TARGET_CHUNK_WORDS and para_words > 100:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_words = para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    # Sisa terakhir
    if current_chunk:
        last_chunk = "\n\n".join(current_chunk)
        # Jika sisa terlalu pendek, gabung dengan chunk sebelumnya
        if current_words < MIN_CHUNK_WORDS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks


def main():
    txt_files = list(EXTRACTED_DIR.glob("*.txt"))

    if not txt_files:
        print("❌ Tidak ada file .txt di data/extracted/")
        print("   Jalankan extract_text.py terlebih dahulu.")
        sys.exit(1)

    print(f"📚 Ditemukan {len(txt_files)} file teks\n")

    total_chunks = 0

    for filepath in txt_files:
        novel_name = filepath.stem
        text = filepath.read_text(encoding="utf-8")
        word_count = len(text.split())

        if word_count < 100:
            print(f"⏭️  {novel_name}: terlalu pendek ({word_count} kata), skip\n")
            continue

        print(f"📖 {novel_name} ({word_count:,} kata)")

        # Coba deteksi bab dulu
        chapters = try_chapter_split(text)
        if chapters:
            print(f"   ✅ Pola bab terdeteksi: {len(chapters)} bab")
        else:
            # Fallback: smart split berdasarkan paragraf
            chapters = smart_split(text)
            print(f"   📐 Smart split: {len(chapters)} potongan (~{TARGET_CHUNK_WORDS} kata/potong)")

        novel_dir = CHAPTERS_DIR / novel_name
        novel_dir.mkdir(parents=True, exist_ok=True)

        # Hapus file lama
        for old_file in novel_dir.glob("*.txt"):
            old_file.unlink()

        for i, content in enumerate(chapters, 1):
            title = f"bab_{i:02d}"
            chapter_file = novel_dir / f"{title}.txt"
            chapter_file.write_text(content, encoding="utf-8")
            cw = len(content.split())
            print(f"   📝 {title}: {cw:,} kata")

        total_chunks += len(chapters)
        print()

    print(f"🎉 Selesai! Total {total_chunks} potongan di data/chapters/")


if __name__ == "__main__":
    main()
