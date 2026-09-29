import logging
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models

from modules.LabelCropper import LabelCropper

logger = logging.getLogger(__name__)

# короткое имя → (модуль, класс, HF-модель); коллекция в Qdrant своя для каждой модели
VECTORIZERS = {
    "siglip2": ("modules.SigLIP2Vectorizer", "SigLIP2Vectorizer", "google/siglip2-base-patch16-384"),
    "dinov2": ("modules.DINOv2Vectorizer", "DINOv2Vectorizer", "facebook/dinov2-base"),
}


def build_vectorizer(name: str, device: Optional[str] = None):
    import importlib

    module, cls, model_name = VECTORIZERS[name]
    return getattr(importlib.import_module(module), cls)(model_name=model_name, device=device)


def collection_name(vectorizer: str) -> str:
    return f"wine_labels_{vectorizer}"


def to_bgr(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """Путь, PIL (RGB) или BGR-массив (как из cv2) → BGR-массив."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, str):
        with Image.open(image) as im:
            image = im.convert("RGB")
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


class LabelSearcher:
    """
    Поиск вина по этикетке: кроп этикетки, ближайшей к центру кадра →
    вектор → ближайшие этикетки каталога в Qdrant.

    Эталоны в Qdrant строятся тем же методом embed (services/label_catalog_indexer.py),
    поэтому запрос и каталог проходят одинаковый кроп и препроцессинг.
    Если этикетка не найдена, векторизуется весь кадр.
    """

    def __init__(
        self,
        vectorizer: str = "siglip2",
        detector_path: str = "models/label_detector.pt",
        qdrant_url: str = "http://127.0.0.1:6333",
        collection: Optional[str] = None,
        device: Optional[str] = None,
        conf_thresh: float = 0.35,
    ):
        self.vectorizer_name = vectorizer
        self.vec = build_vectorizer(vectorizer, device)
        cropper_device = "0" if self.vec.device.type == "cuda" else "cpu"
        self.cropper = LabelCropper(model_path=detector_path, conf_thresh=conf_thresh, device=cropper_device)
        self.client = QdrantClient(url=qdrant_url, timeout=60)
        self.collection = collection or collection_name(vectorizer)

    def crop(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """Кроп этикетки (RGB) и метаданные; при отсутствии этикетки — весь кадр."""
        bgr = to_bgr(image)
        result = self.cropper.crop_array(bgr, return_metadata=True)
        if result is None:
            crop, meta = bgr, {"crop": "full"}
        else:
            crop, meta = result
            meta["crop"] = "label"
        return {"rgb": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), "meta": meta}

    def embed(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict]:
        """Кроп + вектор для пачки изображений: [{'vector', 'meta', 'rgb'}]."""
        crops = [self.crop(i) for i in images]
        vectors = self.vec.vectorize_images([c["rgb"] for c in crops])
        for c, v in zip(crops, vectors):
            c["vector"] = v
        return crops

    def search_vector(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        """Ближайшие вина; у вина может быть несколько эталонов — берётся лучший."""
        res = self.client.query_points_groups(
            self.collection,
            query=vector,
            group_by="slug",
            limit=top_k,
            group_size=1,
            with_payload=["slug", "name", "url"],
        )
        return [
            {"slug": g.id, "name": g.hits[0].payload.get("name"), "url": g.hits[0].payload.get("url"),
             "score": float(g.hits[0].score)}
            for g in res.groups
        ]

    def search(self, image: Union[str, Image.Image, np.ndarray], top_k: int = 5) -> Dict:
        """{'matches': [{'slug', 'name', 'url', 'score'}, ...], 'crop': {...}}, matches по убыванию score."""
        e = self.embed([image])[0]
        return {"matches": self.search_vector(e["vector"], top_k), "crop": e["meta"]}
