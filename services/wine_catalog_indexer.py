"""
Сервис первичного наполнения Qdrant векторами вин из каталога https://vino-svoe.ru/wines/

Запуск (из TriButtonsModels):
    python -m services.wine_catalog_indexer --qdrant-url http://127.0.0.1:6333
Повторный запуск идемпотентен: id точки = uuid5(slug:view), уже проиндексированные вина пропускаются.
"""
import argparse
import io
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models

from modules.SigLIP2Vectorizer import SigLIP2Vectorizer

logger = logging.getLogger("wine_indexer")

BASE_URL = "https://vino-svoe.ru"
CATALOG_URL = f"{BASE_URL}/wines"
SLUG_RE = re.compile(r"^/wines/([^/?#]+)$")
# эталонное фото карточки; другие виды (в руке, на полке...) добавляются позже как отдельные точки
VIEW_CATALOG = "catalog"


@dataclass
class Wine:
    slug: str
    url: str
    name: str
    image_url: str


class CatalogClient:
    def __init__(self, delay: float = 0.5, verify_ssl: bool = True):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (compatible; WineIndexer/1.0)"
        self.s.verify = verify_ssl
        self.delay = delay

    def _get(self, url: str, **kw) -> requests.Response:
        for attempt in range(3):
            try:
                r = self.s.get(url, timeout=30, **kw)
                r.raise_for_status()
                time.sleep(self.delay)
                return r
            except requests.RequestException as e:
                logger.warning("GET %s failed (%s), retry %d", url, e, attempt + 1)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"не удалось загрузить {url}")

    def iter_slugs(self, max_pages: Optional[int] = None) -> Iterator[str]:
        """Обходит ?page=N, пока страницы дают новые slug'и."""
        seen: Set[str] = set()
        page = 1
        while max_pages is None or page <= max_pages:
            html = self._get(CATALOG_URL, params={"page": page}).text
            new = []
            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                m = SLUG_RE.match(a["href"])
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    new.append(m.group(1))
            if not new:
                return
            logger.info("страница %d: +%d вин (всего %d)", page, len(new), len(seen))
            yield from new
            page += 1

    def fetch_wine(self, slug: str) -> Optional[Wine]:
        url = f"{CATALOG_URL}/{slug}"
        soup = BeautifulSoup(self._get(url).text, "html.parser")
        img = soup.select_one("img.wine-hero-block__bottle")
        if img is None or not img.get("src"):
            return None
        h1 = soup.select_one("h1")
        return Wine(slug=slug, url=url, name=h1.get_text(strip=True) if h1 else slug, image_url=img["src"])

    def fetch_image(self, image_url: str) -> Image.Image:
        return Image.open(io.BytesIO(self._get(image_url).content)).convert("RGB")


class WineIndex:
    def __init__(self, url: str, collection: str, dim: int):
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # payload-индексы под группировку/фильтрацию при поиске
            for f in ("slug", "view"):
                self.client.create_payload_index(collection, f, models.PayloadSchemaType.KEYWORD)

    @staticmethod
    def point_id(slug: str, view: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slug}:{view}"))

    def exists(self, point_id: str) -> bool:
        return bool(self.client.retrieve(self.collection, ids=[point_id], with_payload=False, with_vectors=False))

    def upsert(self, wines: List[Wine], vectors: List[List[float]], model_name: str) -> None:
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=self.point_id(w.slug, VIEW_CATALOG),
                    vector=v,
                    payload={"slug": w.slug, "url": w.url, "name": w.name, "image_url": w.image_url,
                             "view": VIEW_CATALOG, "model": model_name},
                )
                for w, v in zip(wines, vectors)
            ],
            wait=True,
        )


def flush(batch, vec, index) -> int:
    if not batch:
        return 0
    index.upsert([w for w, _ in batch], vec.vectorize_images([i for _, i in batch]), vec.model_name)
    n = len(batch)
    logger.info("записано %d векторов", n)
    batch.clear()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="wines_siglip2_b16_384")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--delay", type=float, default=0.5, help="пауза между запросами к сайту, сек")
    ap.add_argument("--no-verify-ssl", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    catalog = CatalogClient(args.delay, verify_ssl=not args.no_verify_ssl)
    vec = SigLIP2Vectorizer(device=args.device)
    index = WineIndex(args.qdrant_url, args.collection, vec.dim)

    batch, done, skipped, failed = [], 0, 0, 0
    for slug in catalog.iter_slugs(args.max_pages):
        if index.exists(index.point_id(slug, VIEW_CATALOG)):
            skipped += 1
            continue
        try:
            wine = catalog.fetch_wine(slug)
            if wine is None:
                raise ValueError("на карточке нет фото бутылки")
            batch.append((wine, catalog.fetch_image(wine.image_url)))
        except Exception as e:
            failed += 1
            logger.error("%s: %s", slug, e)
            continue
        if len(batch) >= args.batch_size:
            done += flush(batch, vec, index)
    done += flush(batch, vec, index)
    logger.info("готово: добавлено %d, пропущено %d, ошибок %d", done, skipped, failed)


if __name__ == "__main__":
    main()
