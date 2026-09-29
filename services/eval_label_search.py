"""
Проверка поиска по этикетке на реальных фото.

С разметкой (TSV с заголовком image_path<TAB>slug, пути относительно --images-dir):
    python -m services.eval_label_search --images-dir ../Dataset/RealPhotos \
        --manifest ./dataset/real_photos_gt.tsv --vectorizer siglip2
печатает top-1/top-5 и пишет predictions.tsv с ошибками.

Без разметки — пишет predictions.tsv (image_path, slug=top-1, score ...),
который удобно исправить руками и дальше использовать как --manifest.

--sheets-dir: картинки «кроп запроса | top-3 кропа каталога» для визуальной
проверки (кропы каталога берутся из --catalog-crops-dir индексатора).
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from modules.LabelSearcher import LabelSearcher

logger = logging.getLogger("eval_label_search")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_manifest(path: Optional[str], images_dir: Path) -> List[Dict]:
    if path is None:
        return [{"image_path": p.name, "slug": ""} for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{"image_path": r["image_path"], "slug": (r.get("slug") or "").strip()}
                for r in csv.DictReader(f, delimiter="\t")]


def save_sheet(query_rgb, matches: List[Dict], catalog_crops: Optional[Path], dst: Path, h: int = 320) -> None:
    tiles = [Image.fromarray(query_rgb)]
    for m in matches[:3]:
        p = catalog_crops / f"{m['slug']}.jpg" if catalog_crops else None
        if p and p.exists():
            tiles.append(Image.open(p).convert("RGB"))
    tiles = [t.resize((max(1, int(t.width * h / t.height)), h)) for t in tiles]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), h), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 10
    sheet.save(dst, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--manifest", help="TSV image_path<TAB>slug; без него — только предсказания")
    ap.add_argument("--vectorizer", choices=["siglip2", "dinov2"], default="siglip2")
    ap.add_argument("--collection")
    ap.add_argument("--detector", default="models/label_detector.pt")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="./predictions.tsv")
    ap.add_argument("--sheets-dir")
    ap.add_argument("--catalog-crops-dir", help="кропы каталога из label_catalog_indexer --crops-dir")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    images_dir = Path(args.images_dir)
    rows = read_manifest(args.manifest, images_dir)
    searcher = LabelSearcher(args.vectorizer, args.detector, args.qdrant_url, args.collection, args.device)
    sheets = Path(args.sheets_dir) if args.sheets_dir else None
    if sheets:
        sheets.mkdir(parents=True, exist_ok=True)
    catalog_crops = Path(args.catalog_crops_dir) if args.catalog_crops_dir else None

    out_rows, top1, topk, labeled, no_label = [], 0, 0, 0, 0
    for r in rows:
        e = searcher.embed([str(images_dir / r["image_path"])])[0]
        matches = searcher.search_vector(e["vector"], args.top_k)
        slugs = [m["slug"] for m in matches]
        no_label += e["meta"]["crop"] == "full"
        rank = slugs.index(r["slug"]) + 1 if r["slug"] in slugs else None
        if r["slug"]:
            labeled += 1
            top1 += rank == 1
            topk += rank is not None
        out_rows.append({
            "image_path": r["image_path"],
            # без разметки колонка slug = предсказание, чтобы файл можно было поправить и использовать как manifest
            "slug": r["slug"] or (slugs[0] if slugs else ""),
            "pred_slug": slugs[0] if slugs else "",
            "score": f"{matches[0]['score']:.4f}" if matches else "",
            "rank_of_true": rank or "",
            "crop": e["meta"]["crop"],
            "top_k": " ".join(f"{m['slug']}:{m['score']:.3f}" for m in matches),
        })
        if sheets:
            save_sheet(e["rgb"], matches, catalog_crops, sheets / f"{Path(r['image_path']).stem}.jpg")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    logger.info("фото: %d, этикетка не найдена: %d → %s", len(rows), no_label, args.out)
    if labeled:
        logger.info("%s: top-1 %.1f%% (%d/%d), top-%d %.1f%%", searcher.collection,
                    100 * top1 / labeled, top1, labeled, args.top_k, 100 * topk / labeled)


if __name__ == "__main__":
    main()
