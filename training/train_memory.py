\
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
from core.patterns import LABEL_SYNONYMS, normalize

MEM_PATH = Path("data/memory.json")

def load_memory() -> Dict[str, Any]:
    if MEM_PATH.exists():
        return json.load(open(MEM_PATH, "r", encoding="utf-8"))
    return {"label_synonyms": LABEL_SYNONYMS.copy(), "vendors": {}}

def save_memory(mem: Dict[str, Any]) -> None:
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mem, open(MEM_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def parse_txt_labels(txt_path: Path) -> List[str]:
    lines = [normalize(x) for x in open(txt_path, "r", encoding="utf-8", errors="ignore").read().splitlines()]
    fields = [x for x in lines if x]
    return fields

def main(images: str, labels: str) -> None:
    mem: Dict[str, Any] = load_memory()
    images_p = Path(images)
    labels_p = Path(labels)

    # gather by basename
    txt_map = {p.stem: p for p in labels_p.glob("*.txt")}
    png_map = {p.stem: p for p in list(images_p.glob("*.png")) + list(images_p.glob("*.jpg")) + list(images_p.glob("*.jpeg"))}

    matched = sorted(set(txt_map.keys()) & set(png_map.keys()))
    for stem in matched:
        fields = parse_txt_labels(txt_map[stem])
        # augment synonyms with what appears in txt (treated as labels to seek)
        for f in fields:
            if f not in mem["label_synonyms"]:
                mem["label_synonyms"][f] = [f]
            else:
                if f not in mem["label_synonyms"][f]:
                    # ensure the key has itself as synonym (idempotent)
                    pass
        # vendor heuristics (just record we saw these fields for this vendor)
        vendor_key = "generic"
        vendors: Dict[str, Any] = mem.setdefault("vendors", {})
        vendors.setdefault(vendor_key, {"fields": {}})
        for f in fields:
            vendors[vendor_key]["fields"].setdefault(f, 0)
            vendors[vendor_key]["fields"][f] += 1

    save_memory(mem)
    print(f"Learned from {len(matched)} pairs. Memory updated at {MEM_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    main(args.images, args.labels)
