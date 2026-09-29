"""
Индексация этикеток из локального датасета Strapi (Dataset/strapi_dataset_photos.csv,
собран Dataset/match_photos.py): для каждого вина берётся оригинал фото из strapi/uploads,
вырезается этикетка (LabelCropper) и вектор пишется в отдельную коллекцию
wine_labels_<модель>_strapi — каталог с сайта (label_catalog_indexer.py) не затрагивается.

Повторный запуск пропускает уже проиндексированные вина; --recreate пересоздаёт коллекцию.

Запуск (из TriButtonyRecognizer):
    python -m services.label_dataset_indexer --vectorizer siglip2 \
        --crops-dir ./dataset/strapi_label_crops
"""
import argparse
import csv
import logging
import uuid
from pathlib import Path
from typing import Dict, List

from PIL import Image
from qdrant_client.http import models

from modules.LabelSearcher import LabelSearcher, collection_name
from services.label_catalog_indexer import ensure_collection

logger = logging.getLogger("label_dataset_indexer")

VIEW_LABEL = "label_strapi"


def point_id(slug: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slug}:{VIEW_LABEL}"))


def read_dataset(path: Path) -> List[Dict]:
    """Одна запись на slug (в CSV строки повторяются)."""
    wines: Dict[str, Dict] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            slug, photo = r["Slug"].strip(), r["Путь к фото"].strip()
            if slug and photo and slug not in wines:
                wines[slug] = {"slug": slug, "name": r["Название вина"].strip(),
                               "winery": r["Винодельня"].strip(), "photo": photo}
    return list(wines.values())


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def flush(batch: List, searcher: LabelSearcher, crops_dir, stats: Dict) -> None:
    if not batch:
        return
    embedded = searcher.embed([img for _, img in batch])
    points = []
    for (wine, _), e in zip(batch, embedded):
        stats[e["meta"]["crop"]] += 1
        if crops_dir:
            Image.fromarray(e["rgb"]).save(crops_dir / f"{wine['slug']}.jpg", quality=90)
        points.append(models.PointStruct(
            id=point_id(wine["slug"]),
            vector=e["vector"],
            payload={"slug": wine["slug"], "name": wine["name"], "winery": wine["winery"],
                     "url": f"https://vino-svoe.ru/wines/{wine['slug']}", "photo": wine["photo"],
                     "view": VIEW_LABEL, "model": searcher.vec.model_name, "crop": e["meta"]},
        ))
    searcher.client.upsert(searcher.collection, points=points, wait=True)
    logger.info("записано %d (этикетка: %d, весь кадр: %d)", len(points), stats["label"], stats["full"])
    batch.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--dataset-dir", default="../Dataset", help="папка с strapi_dataset_photos.csv")
    ap.add_argument("--csv", default="strapi_dataset_photos.csv")
    ap.add_argument("--vectorizer", choices=["siglip2", "dinov2"], default="siglip2")
    ap.add_argument("--collection", help=f"по умолчанию {collection_name('<vectorizer>')}_strapi")
    ap.add_argument("--detector", default="models/label_detector.pt")
    ap.add_argument("--crops-dir", help="сохранять кропы для визуальной проверки")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="ограничить число вин (для тестового прогона)")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    collection = args.collection or f"{collection_name(args.vectorizer)}_strapi"
    searcher = LabelSearcher(args.vectorizer, args.detector, args.qdrant_url, collection, args.device)
    ensure_collection(searcher.client, searcher.collection, searcher.vec.dim, args.recreate)
    dataset_dir = Path(args.dataset_dir)
    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    wines = read_dataset(dataset_dir / args.csv)[: args.limit]
    logger.info("вин в датасете: %d → коллекция %s", len(wines), searcher.collection)
    stats = {"label": 0, "full": 0}
    batch, skipped, failed = [], 0, 0
    for wine in wines:
        if searcher.client.retrieve(searcher.collection, ids=[point_id(wine["slug"])]):
            skipped += 1
            continue
        try:
            batch.append((wine, load_image(dataset_dir / wine["photo"])))
        except Exception as e:
            failed += 1
            logger.error("%s: %s", wine["slug"], e)
            continue
        if len(batch) >= args.batch_size:
            flush(batch, searcher, crops_dir, stats)
    flush(batch, searcher, crops_dir, stats)
    logger.info("готово: этикетка найдена %d, весь кадр %d, пропущено %d, ошибок %d",
                stats["label"], stats["full"], skipped, failed)


if __name__ == "__main__":
    main()
