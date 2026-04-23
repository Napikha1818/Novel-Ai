# 🖊️ Eka Kurniawan Novel Writer — Fine-tuned LLM

Fine-tune Qwen2.5-7B untuk menulis novel bergaya Eka Kurniawan.

## Persyaratan

- Python 3.10+
- GPU NVIDIA L4 (24GB VRAM) atau setara
- CUDA 12.x

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Taruh file novel di `data/raw/`

Letakkan file EPUB dan PDF novel Eka Kurniawan:
- Cantik Itu Luka
- Lelaki Harimau
- Seperti Dendam, Rindu Harus Dibayar Tuntas
- O

### 2. Ekstrak teks

```bash
python scripts/extract_text.py
```

### 3. Potong per bab

```bash
python scripts/split_chapters.py
```

### 4. Buat dataset

```bash
python scripts/build_dataset.py
```

### 5. Fine-tune

```bash
accelerate launch -m axolotl.cli.train training/config.yaml
```

### 6. Menulis novel

```bash
python scripts/generate_novel.py --model output/eka-qwen2.5-7b-qlora
```

## Cara Pakai (Interactive Mode)

```
📝 > /outline Novel tentang keluarga nelayan di pesisir Jawa yang dihantui kutukan leluhur

📝 > /tulis Tuliskan bab 1, perkenalan tokoh utama seorang nelayan bernama Sarwo yang menemukan mayat mengapung di laut. Sekitar 1500 kata.

📝 > /lanjut Lanjutkan ke bab 2, Sarwo pulang dan bertemu istrinya. Ada ketegangan rumah tangga. Masukkan elemen realisme magis.

📝 > /simpan novel_output.txt
```
