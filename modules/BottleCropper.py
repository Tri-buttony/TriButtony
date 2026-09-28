import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from ultralytics import YOLO
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BottleCropper:
    """
    Модуль для детекции и кроппинга бутылок с изображений.
    Возвращает кропы в оперативную память (numpy arrays).
    """
    
    def __init__(
        self, 
        model_path: str = 'yolo26s.pt',
        conf_thresh: float = 0.35,
        # iou_thresh: float = 0.45,
        padding: float = 0.05,
        device: str = 'cpu',
        bottle_class_id: Optional[int] = 39
    ):
        """
        Инициализация кроппера.
        
        Args:
            model_path: Путь к модели YOLO
            conf_thresh: Порог уверенности для детекции
            iou_thresh: Порог IoU для NMS
            padding: Отступ вокруг bbox (в долях от размера bbox, например 0.05 = 5%)
            device: Устройство для инференса ('cpu' или '0' для GPU)
        """
        logger.info(f"Загружаем модель: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        # self.iou_thresh = iou_thresh
        self.padding = padding
        self.device = device
        self.bottle_class_id = bottle_class_id
        
    def crop_image(
        self, 
        image_source: Union[str, Path, io.BytesIO, bytes],
        return_metadata: bool = True
    ) -> Union[Optional[np.ndarray], Optional[Tuple[np.ndarray, dict]]]:
        """
        Находит САМУЮ БОЛЬШУЮ БУТЫЛКУ на изображении и вырезает её.
        Возвращает только один кроп.
        
        Args:
            image_path: Путь к изображению
            return_metadata: Если True, возвращает кортеж (crop, metadata).
                           Если False, возвращает только numpy array (кроп).
        
        Returns:
            Если return_metadata=False:
                np.ndarray - кроп самой большой бутылки, или None если не найдена
            Если return_metadata=True:
                Tuple[np.ndarray, dict] - (кроп, метаданные), или None если не найдена
        """
        # if isinstance(image_path, str):
        #     image_path = Path(image_path)
        #     if not image_path.exists():
        #         raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        #     img = cv2.imread(str(image_path))
        # else:
        #     buffer = image_path
        #     buffer.seek(0)
        #     img_bytes = buffer.read()
        #     img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        #     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
        # # Проверяем, что класс бутылки определен
        # if self.bottle_class_id is None:
        #     logger.error("Класс бутылки не найден в модели. Укажите bottle_class_id при инициализации.")
        #     return None
        
        # # Читаем изображение
        # if img is None:
        #     raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        
        # img_h, img_w = img.shape[:2]
        
        # # Запускаем инференс
        # results = self.model(
        #     str(image_path),
        #     conf=self.conf_thresh,
        #     # iou=self.iou_thresh,
        #     device=self.device,
        #     verbose=False
        # )

        # 1. Универсальное чтение изображения в numpy-массив
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Изображение не найдено: {path}")
            img = cv2.imread(str(path))
            source_info = str(path)
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            img_bytes = image_source.read()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            source_info = "<BytesIO buffer>"
        elif isinstance(image_source, bytes):
            img_array = np.frombuffer(image_source, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            source_info = "<bytes>"
        else:
            raise TypeError(f"Неподдерживаемый тип источника: {type(image_source)}")
    
        if img is None:
            raise ValueError(f"Не удалось декодировать изображение из: {source_info}")
    
        # 2. Проверка класса бутылки
        if self.bottle_class_id is None:
            logger.error("Класс бутылки не найден в модели. Укажите bottle_class_id при инициализации.")
            return None
    
        img_h, img_w = img.shape[:2]
    
        # 3. Инференс: передаём numpy-массив, а не путь/буфер
        #    YOLOv8 умеет принимать np.ndarray напрямую
        results = self.model(
            img,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )
        
        # Ищем САМУЮ БОЛЬШУЮ бутылку
        largest_bottle = None
        largest_area = 0
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                
                # Фильтруем ТОЛЬКО бутылки
                if cls != self.bottle_class_id:
                    continue
                
                # Получаем координаты
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy
                
                # Считаем площадь
                area = (x_max - x_min) * (y_max - y_min)
                
                # Запоминаем самую большую
                if area > largest_area:
                    largest_area = area
                    largest_bottle = {
                        'xyxy': xyxy,
                        'class': cls,
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'area': area
                    }
        
        # Если бутылка не найдена
        if largest_bottle is None:
            logger.info(f"Бутылка не найдена на изображении")
            return None
        
        # Вырезаем САМУЮ БОЛЬШУЮ бутылку
        x_min, y_min, x_max, y_max = largest_bottle['xyxy']
        
        # Добавляем padding
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = box_w * self.padding
        pad_y = box_h * self.padding
        
        x_min_pad = max(0, int(x_min - pad_x))
        y_min_pad = max(0, int(y_min - pad_y))
        x_max_pad = min(img_w, int(x_max + pad_x))
        y_max_pad = min(img_h, int(y_max + pad_y))
        
        # Вырезаем кроп
        crop = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad].copy()
        
        class_name = self.model.names.get(largest_bottle['class'], f"class_{largest_bottle['class']}")
        
        logger.info(
            f"Вырезана самая большая бутылка: "
            f"'{class_name}', уверенность {largest_bottle['confidence']:.2%}, "
            f"площадь {largest_bottle['area']:.0f} px²"
        )
        
        if return_metadata:
            metadata = {
                'bbox': [x_min_pad, y_min_pad, x_max_pad, y_max_pad],
                'class': largest_bottle['class'],
                'confidence': largest_bottle['confidence'],
                'class_name': class_name,
                'original_bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'area': largest_bottle['area']
            }
            return crop, metadata
        else:
            return crop
    
    def crop_and_save(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = 'bottle'
    ) -> List[Path]:
        """
        Находит бутылки, вырезает их и сохраняет на диск.
        
        Args:
            image_path: Путь к изображению
            output_dir: Директория для сохранения кропов
            prefix: Префикс для имен файлов
        
        Returns:
            Список путей к сохраненным файлам
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crops_data = self.crop_image(image_path, return_metadata=True)
        saved_paths = []
        
        image_stem = Path(image_path).stem

        # Теперь crops_data это либо None, либо (crop, metadata)
        if crops_data is None:
            logger.info(f"Бутылка не найдена, нечего сохранять для {Path(image_path).name}")
            return []
        
        crop, data = crops_data
        cls = data['class']
        conf = data['confidence']
        
        # Убираем цикл, сохраняем только ОДИН кроп
        idx = 0
        
        logger.info(f"Сохранено {len(saved_paths)} кропов в {output_dir}")
        return saved_paths
    
    def crop_batch(
        self,
        image_paths: List[Union[str, Path]],
        return_metadata: bool = True
    ) -> Dict[Path, Union[List[np.ndarray], List[Dict]]]:
        """
        Обрабатывает пакет изображений.
        
        Args:
            image_paths: Список путей к изображениям
            return_metadata: Возвращать ли метаданные
        
        Returns:
            Словарь {image_path: crops}
        """
        results = {}
        
        for img_path in image_paths:
            try:
                crop = self.crop_image(img_path, return_metadata=return_metadata)
                results[Path(img_path)] = crop
            except Exception as e:
                logger.error(f"Ошибка при обработке {img_path}: {e}")
                results[Path(img_path)] = None
        
        return results


# ==========================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ==========================================

if __name__ == "__main__":
    # Пример 1: Простой кроппинг с метаданными
    cropper = BottleCropper(
        model_path='yolo11n.pt',
        conf_thresh=0.25,
        padding=0.1  # 10% отступ вокруг bbox
    )
    
    # Обрабатываем одно изображение
    image_path = "path/to/your/image.jpg"
    crops = cropper.crop_image(image_path, return_metadata=True)
    
    print(f"\nНайдено {len(crops)} бутылок:")
    for i, data in enumerate(crops):
        print(f"  Бутылка #{i+1}:")
        print(f"    Класс: {data['class_name']} (ID: {data['class']})")
        print(f"    Уверенность: {data['confidence']:.2%}")
        print(f"    Размер кропа: {data['crop'].shape}")
        print(f"    Координаты: {data['bbox']}")
    
    # Пример 2: Только кропы (без метаданных)
    crops_only = cropper.crop_image(image_path, return_metadata=False)
    print(f"\nПолучено {len(crops_only)} numpy arrays")
    
    # Пример 3: Сохранение на диск
    output_dir = "cropped_bottles"
    saved_files = cropper.crop_and_save(image_path, output_dir, prefix='wine')
    print(f"\nСохранено файлов: {len(saved_files)}")
    
    # Пример 4: Пакетная обработка
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    batch_results = cropper.crop_batch(image_paths, return_metadata=True)
    
    for img_path, crops in batch_results.items():
        print(f"{img_path.name}: {len(crops)} бутылок")