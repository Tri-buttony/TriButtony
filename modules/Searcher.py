from modules.ImageVectorDB import ImageVectorDB
from modules.BottleCropper import BottleCropper
from modules.SigLIP2Vectorizer import SigLIP2Vectorizer

import io
from PIL import Image
import PIL

vectorizer = SigLIP2Vectorizer()
cropper = BottleCropper(model_path='models/yolo26s.pt', bottle_class_id=39, conf_thresh=0.35)
db = ImageVectorDB(collection_name="wines_siglip2_b16_384")

def vectorize(images_batch: PIL.Image):
    vec = vectorizer.vectorize_images([image for image in images_batch])
    return vec

def crop(image_path: str):
    bottle = cropper.crop_image(image_path=image_path)
    return bottle


# Тут если что надо поменять, чтобы можно было читать из буфера например
def search_similar(image_path, crop=False):
    if crop:
        crop_bottle = crop(image_path)
        if not crop_bottle:
            return None
        crop_bottle = crop_bottle[0]
        bottle = Image.fromarray(crop_bottle)
    else:
        bottle = Image.open(image_path)
        
    vec = vectorize([bottle])[0]
    results = db.search_similar(vec)
    
    return results

