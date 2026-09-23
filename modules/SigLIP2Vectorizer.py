import numpy as np
import torch
from PIL import Image
from typing import List, Union
from transformers import AutoModel, AutoProcessor


class SigLIP2Vectorizer:
    """Эмбеддер изображений на google/siglip2-base-patch16-384 (dim=768, L2-нормализованные вектора)."""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-384", device: str = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.dim = int(self.model.config.vision_config.hidden_size)

    @torch.no_grad()
    def vectorize_images(self, images: List[Union[Image.Image, np.ndarray]]) -> List[List[float]]:
        pil = [Image.fromarray(i) if isinstance(i, np.ndarray) else i for i in images]
        pil = [i.convert("RGB") for i in pil]
        inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):  # новые версии transformers могут вернуть ModelOutput
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().tolist()

    def vectorize_array(self, crop: np.ndarray) -> List[float]:
        return self.vectorize_images([crop])[0]
