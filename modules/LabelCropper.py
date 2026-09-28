import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class LabelCropper:
    """
    Детекция этикеток и кроп той, что ближе всего к центру изображения.

    Модель — одноклассовый детектор `label`, обученный
    services/train_label_detector.py.
    """

    def __init__(
        self,
        model_path: str = 'models/label_detector.pt',
        conf_thresh: float = 0.35,
        padding: float = 0.03,
        device: str = 'cpu',
        imgsz: int = 640,
    ):
        """
        Args:
            model_path: Путь к весам детектора этикеток
            conf_thresh: Порог уверенности для детекции
            padding: Отступ вокруг bbox (в долях от размера bbox)
            device: 'cpu' или '0' для GPU
            imgsz: Размер входа модели
        """
        logger.info(f"Загружаем модель: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.padding = padding
        self.device = device
        self.imgsz = imgsz

    def detect(self, img: np.ndarray) -> List[Dict]:
        """Все этикетки на BGR-изображении, отсортированные по близости к центру."""
        img_h, img_w = img.shape[:2]
        results = self.model(img, conf=self.conf_thresh, imgsz=self.imgsz, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        # расстояние центра bbox до центра кадра, в долях диагонали
        dist = np.hypot(cx - img_w / 2, cy - img_h / 2) / np.hypot(img_w, img_h)

        dets = [
            {
                'xyxy': xyxy[i],
                'confidence': float(conf[i]),
                'center_dist': float(dist[i]),
                'area': float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
            }
            for i in range(len(xyxy))
        ]
        dets.sort(key=lambda d: (d['center_dist'], -d['confidence']))
        return dets

    def crop_array(
        self,
        img: np.ndarray,
        return_metadata: bool = True,
    ) -> Union[Optional[np.ndarray], Optional[Tuple[np.ndarray, dict]]]:
        """
        Вырезает этикетку, ближайшую к центру BGR-изображения.

        Returns:
            None, если этикетка не найдена; иначе кроп или (кроп, метаданные).
        """
        dets = self.detect(img)
        if not dets:
            return None

        img_h, img_w = img.shape[:2]
        best = dets[0]
        x_min, y_min, x_max, y_max = best['xyxy']
        pad_x = (x_max - x_min) * self.padding
        pad_y = (y_max - y_min) * self.padding
        x0 = max(0, int(x_min - pad_x))
        y0 = max(0, int(y_min - pad_y))
        x1 = min(img_w, int(x_max + pad_x))
        y1 = min(img_h, int(y_max + pad_y))
        crop = img[y0:y1, x0:x1].copy()

        if not return_metadata:
            return crop
        metadata = {
            'bbox': [x0, y0, x1, y1],
            'original_bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'confidence': best['confidence'],
            'center_dist': best['center_dist'],
            'area': best['area'],
            'num_labels': len(dets),
        }
        return crop, metadata

    def crop_image(
        self,
        image_path: Union[str, Path],
        return_metadata: bool = True,
    ) -> Union[Optional[np.ndarray], Optional[Tuple[np.ndarray, dict]]]:
        """То же, что crop_array, но по пути к файлу."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        result = self.crop_array(img, return_metadata=return_metadata)
        if result is None:
            logger.info(f"Этикетка не найдена на изображении {image_path.name}")
        return result


if __name__ == "__main__":
    # Визуальная проверка: python -m modules.LabelCropper <папка с фото> <папка вывода> [веса]
    import sys

    logging.basicConfig(level=logging.INFO)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cropper = LabelCropper(model_path=sys.argv[3] if len(sys.argv) > 3 else 'models/label_detector.pt', device='0')
    (dst / 'crops').mkdir(parents=True, exist_ok=True)
    (dst / 'debug').mkdir(parents=True, exist_ok=True)

    found = 0
    paths = [p for p in sorted(src.iterdir()) if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = cropper.detect(img)
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = map(int, d['xyxy'])
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
            cv2.putText(img, f"{d['confidence']:.2f}", (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(dst / 'debug' / f"{p.stem}.jpg"), img)
        result = cropper.crop_image(p, return_metadata=False)
        if result is not None:
            found += 1
            cv2.imwrite(str(dst / 'crops' / f"{p.stem}.jpg"), result)
    print(f"Этикетка найдена на {found}/{len(paths)} фото → {dst}")
