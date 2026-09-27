import sys
import os
import io
import base64
import json

import matplotlib.pyplot as plt
import cv2
from PIL import Image

from openai import OpenAI

from modules.BottleCropper import BottleCropper
from modules.SerchVinoSvoe import WineSearcher, OCRData

from config import (
    LOCAL_LLM_PATH, 
    TOP_K_SEARCH, 
    SYSTEM_PROMPT, 
    USER_PROMPT, 
    CROPPER_PATH, 
    CROPPER_TRESHOLD, 
    MAX_CANDIDATES, 
    TEMPERATURE, 
    TOP_P,
    SEARCH_SCORE_TRESHOLD
)

def _llm_call(client, system_prompt="Ты ополезный помощьник и твечаешь кратко и по делу", user_prompt="Привет", image_b64=None):

    messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                ]
            }
        ]
    
    if image_b64:
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", "content": [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
        
    # print(messages)
    
    r = client.chat.completions.create(model="gemma", messages=messages, temperature=TEMPERATURE, top_p=TOP_P)
    return r.choices[0].message.content

TOP_K = TOP_K_SEARCH

client = OpenAI(base_url=LOCAL_LLM_PATH, api_key="none")

try:
    ans = _llm_call(client, user_prompt="Привет")
except Exception as e:
    print("Тестовый запрос к LLM не удался. Проверьде доступность модели")
    raise(e)


cropper = BottleCropper(model_path=CROPPER_PATH, bottle_class_id=0, conf_thresh=CROPPER_TRESHOLD)
searcher = WineSearcher(max_candidates=MAX_CANDIDATES)

def search_by_ocr(images: list[str]):
    res = []
    for image_path in images:
        print(f"Обрабатываем {image_path}")
        # 1. Получаем кроп бутылки
        cropped_array = cropper.crop_image(image_path)[0]
        cropped_image = Image.fromarray(cropped_array)
            
        # 2. Сохраняем кроп в буфер и кодируем в base64
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 3. Отдаем в LLM base64 именно кропа, а не исходного изображения
        ocr_raw = _llm_call(
            client, 
            system_prompt=SYSTEM_PROMPT, 
            user_prompt=USER_PROMPT, 
            image_b64=b64
        )

        print("LLM извлекла")
        print(ocr_raw)
        ocr_json = ocr_raw.replace("json", "").replace("```", "")

        print(ocr_json)
        # 4. Парсим ответ
        try:
            ocr_data = OCRData(**json.loads(ocr_json))
        except Exception as e:
            print(f"Ошибка в json для {image_path}: {e}")
            continue  # Пропускаем итерацию, чтобы избежать NameError на шаге поиска
            
        # 5. Ищем результаты
        results = searcher.search(
            ocr_data,
            top_k=TOP_K,
        )
        res.append(results)
        
    return res