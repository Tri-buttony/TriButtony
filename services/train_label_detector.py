"""
Обучение одноклассового детектора этикеток на датасете из
label_dataset_builder.py. Лучшие веса копируются в models/label_detector.pt
(их по умолчанию грузит modules/LabelCropper.py).

Запуск (из TriButtonsModels):
    python -m services.train_label_detector --data ./dataset/labels_yolo/data.yaml

На GTX 1650 Ti (4 ГБ) yolo26s с imgsz=640 помещается при batch 8;
при OOM уменьшайте --batch или берите yolo26n.pt.
"""
import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger("train_label_detector")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset/labels_yolo/data.yaml")
    ap.add_argument("--model", default="yolo26s.pt", help="предобученные COCO-веса для дообучения")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="./runs/label_detector")
    ap.add_argument("--name", default="train")
    ap.add_argument("--out", default="./models/label_detector.pt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        single_cls=True,
        patience=20,
        # этикетка бывает у края кадра и повёрнута; mosaic отключаем под конец,
        # чтобы модель доучилась на цельных кадрах, как в инференсе
        degrees=10.0,
        close_mosaic=10,
        fliplr=0.0,  # зеркальный текст на этикетках не встречается
    )

    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    logger.info("val: mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f",
                metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    logger.info("Веса: %s → %s", best, out)


if __name__ == "__main__":
    main()
