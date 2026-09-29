import numpy as np
import torch
from PIL import Image
from typing import List, Union
from transformers import AutoImageProcessor, AutoModel


class DINOv2Vectorizer:
    """
    Эмбеддер изображений на DINOv2 (L2-нормализованные вектора, dim=768 для base).
    Интерфейс как у SigLIP2Vectorizer.

    Стандартный препроцессинг DINOv2 обрезает центр кадра, а у этикеток
    важные края (название, год) — поэтому кадр просто сжимается в квадрат,
    как у SigLIP2.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None, size: int = 336):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = AutoImageProcessor.from_pretrained(
            model_name, do_center_crop=False, size={"height": size, "width": size}
        )
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def vectorize_images(self, images: List[Union[Image.Image, np.ndarray]]) -> List[List[float]]:
        pil = [Image.fromarray(i) if isinstance(i, np.ndarray) else i for i in images]
        pil = [i.convert("RGB") for i in pil]
        inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
        feats = self.model(**inputs).pooler_output  # CLS-токен после layernorm
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().tolist()

    def vectorize_array(self, crop: np.ndarray) -> List[float]:
        return self.vectorize_images([crop])[0]
