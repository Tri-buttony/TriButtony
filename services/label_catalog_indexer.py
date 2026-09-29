"""
Индексация этикеток каталога: для каждого вина из коллекции каталога
(wine_catalog_indexer.py) берётся фото карточки, вырезается этикетка
(LabelCropper) и её вектор пишется в отдельную коллекцию wine_labels_<модель>.

Фото кэшируются в --images-dir, поэтому переиндексация другой моделью
не качает каталог заново. Повторный запуск пропускает уже проиндексированные
вина; --recreate пересоздаёт коллекцию (нужно после переобучения детектора).

Запуск (из TriButtonyRecognizer):
    python -m services.label_catalog_indexer --no-verify-ssl --vectorizer siglip2 \
        --crops-dir ./dataset/catalog_label_crops
"""
import argparse
import logging
import uuid
from pathlib import Path
from typing import Dict, Iterator, List

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models

from modules.LabelSearcher import LabelSearcher, collection_name
from services.wine_catalog_indexer import CatalogClient

logger = logging.getLogger("label_catalog_indexer")

VIEW_LABEL = "label_catalog"


def iter_catalog(client: QdrantClient, collection: str) -> Iterator[Dict]:
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            yield p.payload
        if offset is None:
            return


def point_id(slug: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slug}:{VIEW_LABEL}"))


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        # slug — ключ группировки в LabelSearcher.search_vector
        for f in ("slug", "view"):
            client.create_payload_index(name, f, models.PayloadSchemaType.KEYWORD)


def load_image(wine: Dict, catalog: CatalogClient, images_dir: Path) -> Image.Image:
    path = images_dir / f"{wine['slug']}{Path(wine['image_url']).suffix or '.img'}"
    if not path.exists():
        path.write_bytes(catalog.fetch_bytes(wine["image_url"]))
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
            payload={"slug": wine["slug"], "name": wine.get("name"), "url": wine.get("url"),
                     "image_url": wine["image_url"], "view": VIEW_LABEL, "model": searcher.vec.model_name,
                     "crop": e["meta"]},
        ))
    searcher.client.upsert(searcher.collection, points=points, wait=True)
    logger.info("записано %d (этикетка: %d, весь кадр: %d)", len(points), stats["label"], stats["full"])
    batch.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--catalog-collection", default="wines_siglip2_b16_384", help="откуда брать список вин")
    ap.add_argument("--vectorizer", choices=["siglip2", "dinov2"], default="siglip2")
    ap.add_argument("--collection", help=f"по умолчанию {collection_name('<vectorizer>')}")
    ap.add_argument("--detector", default="models/label_detector.pt")
    ap.add_argument("--images-dir", default="./dataset/catalog_images")
    ap.add_argument("--crops-dir", help="сохранять кропы для визуальной проверки")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="ограничить число вин (для тестового прогона)")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--delay", type=float, default=0.3, help="пауза между запросами к сайту, сек")
    ap.add_argument("--no-verify-ssl", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    searcher = LabelSearcher(args.vectorizer, args.detector, args.qdrant_url, args.collection, args.device)
    ensure_collection(searcher.client, searcher.collection, searcher.vec.dim, args.recreate)
    catalog = CatalogClient(args.delay, verify_ssl=not args.no_verify_ssl)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    wines = list(iter_catalog(searcher.client, args.catalog_collection))[: args.limit]
    logger.info("вин в каталоге: %d → коллекция %s", len(wines), searcher.collection)
    stats = {"label": 0, "full": 0}
    batch, skipped, failed = [], 0, 0
    for wine in wines:
        if searcher.client.retrieve(searcher.collection, ids=[point_id(wine["slug"])]):
            skipped += 1
            continue
        try:
            batch.append((wine, load_image(wine, catalog, images_dir)))
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
