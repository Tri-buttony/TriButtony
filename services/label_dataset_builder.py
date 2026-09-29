"""
Сборка датасета для детектора этикеток (один класс `label`) из публичных
наборов Roboflow Universe.

Что делает:
1. Для каждого набора берёт последнюю версию (или указанную через
   `workspace/project:N`) и скачивает экспорт в формате YOLO через REST API.
   Нужен API-ключ Roboflow: --api-key или переменная ROBOFLOW_API_KEY.
2. Сводит классы к одному `label` по правилу набора (--rule):
       all          — каждый бокс любого класса становится `label` (по умолчанию)
       keep:a,b     — только боксы классов a, b
       union        — все боксы кадра сливаются в один (для наборов, где
                      размечены поля этикетки, а не этикетка целиком)
   Полигоны (экспорт сегментационных проектов) переводятся в bbox.
3. Заново делит на train/val по исходным снимкам: аугментированные копии
   Roboflow (`<stem>_jpg.rf.<hash>.jpg`) одного снимка попадают в один сплит,
   иначе val «подсматривает» в train. Почти одинаковые кадры из разных
   наборов (форки) отсеиваются по phash.

Сначала стоит посмотреть классы:
    python -m services.label_dataset_builder --inspect

Сборка (из TriButtonsModels):
    python -m services.label_dataset_builder --out-dir ./dataset/labels_yolo \
        --rule vinokroboflow/wine-label-detection-ti2rc=keep:label
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import requests
import yaml
from PIL import Image

logger = logging.getLogger("label_dataset_builder")

API = "https://api.roboflow.com"
DEFAULT_DATASETS = [
    "product-and-label-analysis/label-detection-u9cof",
    "vinokroboflow/wine-label-detection-ti2rc",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")

Box = Tuple[float, float, float, float]  # cx, cy, w, h (нормированные)


@dataclass
class Sample:
    dataset: str
    group: str       # исходный снимок — единица сплита
    image: Path
    boxes: List[Box]


# ---------- Roboflow ----------

def http_get(url: str, params: Optional[dict] = None, **kw) -> requests.Response:
    """GET, чьи ошибки не раскрывают ключ: api_key живёт в query-строке, а
    requests печатает полный URL в тексте исключения и трейсбеке."""
    try:
        r = requests.get(url, params=params, **kw)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        status = e.response.status_code if e.response is not None else type(e).__name__
        raise RuntimeError(f"GET {url.split('?', 1)[0]}: {status}") from None

def resolve_version(spec: str, api_key: str) -> Tuple[str, str, int]:
    ref, _, ver = spec.partition(":")
    ws, proj = ref.split("/")
    if ver:
        return ws, proj, int(ver)
    r = http_get(f"{API}/{ws}/{proj}", params={"api_key": api_key}, timeout=60)
    versions = r.json().get("versions") or []
    if not versions:
        raise RuntimeError(f"{ref}: нет опубликованных версий")
    latest = max(int(v["id"].rsplit("/", 1)[-1]) for v in versions)
    return ws, proj, latest


def download(spec: str, api_key: str, cache: Path, fmt: str) -> Path:
    ws, proj, ver = resolve_version(spec, api_key)
    dst = cache / f"{ws}__{proj}__v{ver}__{fmt}"
    if (dst / "data.yaml").exists():
        logger.info("%s v%d: уже скачан", spec, ver)
        return dst
    r = http_get(f"{API}/{ws}/{proj}/{ver}/{fmt}", params={"api_key": api_key}, timeout=120)
    link = r.json()["export"]["link"]
    logger.info("%s v%d: скачиваю экспорт %s", spec, ver, fmt)
    zpath = cache / f"{dst.name}.zip"
    with http_get(link, stream=True, timeout=600) as resp:
        with open(zpath, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    zpath.unlink()
    return dst


# ---------- разбор экспорта ----------

def parse_label_line(parts: List[str]) -> Tuple[int, Box]:
    cls, vals = int(parts[0]), list(map(float, parts[1:]))
    if len(vals) == 4:
        return cls, tuple(vals)
    xs, ys = vals[0::2], vals[1::2]  # полигон → bbox
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return cls, ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def union_box(boxes: List[Box]) -> Box:
    x0 = min(b[0] - b[2] / 2 for b in boxes)
    y0 = min(b[1] - b[3] / 2 for b in boxes)
    x1 = max(b[0] + b[2] / 2 for b in boxes)
    y1 = max(b[1] + b[3] / 2 for b in boxes)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def read_names(root: Path) -> List[str]:
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def iter_pairs(root: Path):
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS or img.parent.name != "images":
            continue
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        yield img, lbl


def load_dataset(spec: str, root: Path, rule: str) -> Tuple[List[Sample], Counter]:
    names = read_names(root)
    keep: Optional[set] = None
    if rule.startswith("keep:"):
        keep = {c.strip() for c in rule[5:].split(",")}
        unknown = keep - set(names)
        if unknown:
            raise ValueError(f"{spec}: нет классов {sorted(unknown)}, есть {names}")
    stats: Counter = Counter()
    samples = []
    for img, lbl in iter_pairs(root):
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, box = parse_label_line(parts)
                name = names[cls]
                stats[name] += 1
                if keep is None or name in keep:
                    boxes.append(box)
        if rule == "union" and boxes:
            boxes = [union_box(boxes)]
        group = RF_SUFFIX.sub("", img.stem)
        samples.append(Sample(spec, f"{spec}::{group}", img, boxes))
    return samples, stats


# ---------- сборка ----------

def dedup(samples: List[Sample], hamming: int) -> List[Sample]:
    """Убирает почти одинаковые кадры из РАЗНЫХ наборов (форки на Roboflow)."""
    bits = np.zeros((len(samples), 64), dtype=bool)
    owners = np.empty(len(samples), dtype=object)
    n = 0
    out, dropped = [], 0
    for s in samples:
        with Image.open(s.image) as im:
            h = imagehash.phash(im.convert("RGB")).hash.ravel()
        dist = np.count_nonzero(bits[:n] != h, axis=1)
        if np.any((dist <= hamming) & (owners[:n] != s.dataset)):
            dropped += 1
            continue
        bits[n], owners[n] = h, s.dataset
        n += 1
        out.append(s)
    logger.info("phash-дубликаты между наборами: убрано %d", dropped)
    return out


def split_groups(samples: List[Sample], val_frac: float, seed: int) -> Dict[str, str]:
    groups = sorted({s.group for s in samples})
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_frac))
    return {g: ("val" if i < n_val else "train") for i, g in enumerate(groups)}


def write(samples: List[Sample], out: Path, val_frac: float, seed: int) -> Counter:
    if out.exists():
        shutil.rmtree(out)
    split_of = split_groups(samples, val_frac, seed)
    counts: Counter = Counter()
    for s in samples:
        split = split_of[s.group]
        name = hashlib.md5(f"{s.dataset}/{s.image.name}".encode()).hexdigest()[:16]
        img_dst = out / "images" / split / f"{name}{s.image.suffix.lower()}"
        lbl_dst = out / "labels" / split / f"{name}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.image, img_dst)
        lbl_dst.write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in s.boxes))
        counts[split] += 1
        counts[f"{split}_boxes"] += len(s.boxes)
        counts[f"{split}_empty"] += not s.boxes
    (out / "data.yaml").write_text(
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                        "names": {0: "label"}}, allow_unicode=True),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                    help="workspace/project или workspace/project:версия")
    ap.add_argument("--rule", action="append", default=[],
                    help="workspace/project=all|union|keep:a,b (можно несколько раз)")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--format", default="yolov8", help="формат экспорта Roboflow")
    ap.add_argument("--cache-dir", default="./dataset/roboflow_cache")
    ap.add_argument("--out-dir", default="./dataset/labels_yolo")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-hamming", type=int, default=4, help="-1 — не дедуплицировать")
    ap.add_argument("--inspect", action="store_true", help="только скачать и показать классы")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        ap.error("нужен API-ключ Roboflow: --api-key или ROBOFLOW_API_KEY")
    rules = dict(r.split("=", 1) for r in args.rule)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    all_samples: List[Sample] = []
    report = {}
    for spec in args.datasets:
        root = download(spec, args.api_key, cache, args.format)
        ref = spec.split(":")[0]
        rule = rules.get(ref, "all")
        samples, stats = load_dataset(ref, root, rule)
        n_groups = len({s.group for s in samples})
        logger.info("%s: %d изображений (%d исходных снимков), правило '%s', классы %s",
                    ref, len(samples), n_groups, rule, dict(stats.most_common()))
        report[ref] = {"images": len(samples), "groups": n_groups, "rule": rule, "classes": dict(stats)}
        all_samples.extend(samples)

    if args.inspect:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.dedup_hamming >= 0:
        all_samples = dedup(all_samples, args.dedup_hamming)
    out = Path(args.out_dir)
    counts = write(all_samples, out, args.val_frac, args.seed)
    report["_result"] = dict(counts)
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Готово: %s → %s", dict(counts), out / "data.yaml")


if __name__ == "__main__":
    main()
