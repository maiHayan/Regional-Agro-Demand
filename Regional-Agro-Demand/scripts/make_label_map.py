import json
import numpy as np
from pathlib import Path

p = Path("models/cluster_label_map.json")
out = {}

if not p.exists():
    raise SystemExit(f"JSON not found: {p}")

with p.open("r", encoding="utf-8-sig") as f:
    parsed = json.load(f)

for k, v in parsed.items():
    try:
        out[int(k)] = v
    except Exception:
        out[str(k).strip()] = v

np.save("models/label_map.npy", out, allow_pickle=True)
print("Saved models/label_map.npy with keys:", list(out.keys()))
