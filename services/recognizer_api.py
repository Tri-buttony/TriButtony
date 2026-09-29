"""
HTTP-сервис распознавания вина по фото: кроп этикетки, ближайшей к центру →
вектор → ближайшие этикетки каталога в Qdrant (modules/LabelSearcher.py).

POST /predict  (multipart, поле `image`) →
    [{"slug", "name", "url", "score"}, ...]  — top-k по убыванию score;
    gateway берёт первый элемент как Top-1.
    Заголовок X-Crop: label | full (этикетка не найдена — искали по всему кадру).

Запуск (из TriButtonyRecognizer):
    python -m services.recognizer_api --port 9000
Шлюз:
    RECOGNIZER_URL=http://127.0.0.1:9000/predict uvicorn main:app --port 8080
"""
import argparse
import asyncio
import io
import logging

import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from modules.LabelSearcher import LabelSearcher

logger = logging.getLogger("recognizer_api")


def create_app(searcher: LabelSearcher, top_k: int) -> FastAPI:
    app = FastAPI(title="Wine label recognizer")
    lock = asyncio.Lock()  # одна модель на GPU — запросы по очереди

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "collection": searcher.collection}

    @app.post("/predict")
    async def predict(response: Response, image: UploadFile = File(...)) -> list:
        content = await image.read()
        try:
            with Image.open(io.BytesIO(content)) as im:
                pil = im.convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "not an image")
        async with lock:
            result = await asyncio.to_thread(searcher.search, pil, top_k)
        response.headers["X-Crop"] = result["crop"]["crop"]
        if not result["matches"]:
            raise HTTPException(404, "no matches")
        return result["matches"]

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--vectorizer", choices=["siglip2", "dinov2"], default="siglip2")
    ap.add_argument("--collection")
    ap.add_argument("--detector", default="models/label_detector.pt")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    searcher = LabelSearcher(args.vectorizer, args.detector, args.qdrant_url, args.collection, args.device)
    uvicorn.run(create_app(searcher, args.top_k), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
