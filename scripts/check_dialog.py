import json

data = [json.loads(l) for l in open("data/dataset/train.jsonl", "r", encoding="utf-8")]
total = len(data)

has_dialog = 0
for d in data:
    out = d["output"]
    if '"' in out or "\u201c" in out or "\u201d" in out:
        has_dialog += 1

print(f"Total pairs     : {total}")
print(f"Mengandung dialog: {has_dialog} ({has_dialog/total*100:.0f}%)")

# Cek per tipe
from collections import Counter
type_dialog = Counter()
type_total = Counter()
for d in data:
    t = d["metadata"]["type"]
    type_total[t] += 1
    out = d["output"]
    if '"' in out or "\u201c" in out or "\u201d" in out:
        type_dialog[t] += 1

print("\nPer tipe:")
for t in sorted(type_total.keys()):
    td = type_dialog.get(t, 0)
    tt = type_total[t]
    print(f"  {t:25s} {td:>4d}/{tt:<4d} ({td/tt*100:.0f}%)")
