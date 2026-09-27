import os
import base64
import io
from typing import Optional, Dict, Any, Union

from openai import OpenAI
from PIL import Image


class OpenRouterInference:
    """
    Класс для инференса через OpenRouter API (бесплатные модели).
    Поддерживает текстовые и мультимодальные запросы (изображения).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "meta-llama/llama-4-maverick:free",  # одна из бесплатных моделей
        default_max_tokens: int = 512,
        default_temperature: float = 0.3,
        default_top_p: float = 0.9,
    ):
        """
        Args:
            api_key: API-ключ OpenRouter. Если не указан, ищет в переменной OPENROUTER_API_KEY.
            base_url: Базовый URL для API (по умолчанию OpenRouter).
            model: Имя модели (можно поменять позже в generate).
            default_max_tokens: Максимальное число токенов в ответе.
            default_temperature: Температура (0-1).
            default_top_p: Top-p sampling.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Не указан API-ключ. Передайте его в конструктор или установите "
                "переменную окружения OPENROUTER_API_KEY."
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )
        self.default_model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p

    def _encode_image(self, image_input: Union[str, bytes, Image.Image]) -> str:
        """
        Преобразует изображение в base64-строку (формат JPEG).
        """
        if isinstance(image_input, str):
            # Предполагаем, что это путь к файлу
            with open(image_input, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_input, bytes):
            img_bytes = image_input
        elif isinstance(image_input, Image.Image):
            # Сохраняем в bytes
            buffer = io.BytesIO()
            image_input.convert("RGB").save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
        else:
            raise TypeError("image_input должен быть str (путь), bytes или PIL.Image")

        return base64.b64encode(img_bytes).decode("utf-8")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        image_input: Optional[Union[str, bytes, Image.Image]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Отправляет запрос к OpenRouter и возвращает ответ модели.

        Args:
            user_prompt: Основной текст запроса.
            system_prompt: Системный промпт (необязательно).
            image_input: Изображение (путь, байты или PIL.Image).
            model: Имя модели (переопределяет дефолтную).
            max_tokens: Максимум токенов в ответе.
            temperature: Температура.
            top_p: Top-p.
            **kwargs: Дополнительные параметры для chat.completions.create.

        Returns:
            str: Текст ответа модели.
        """
        # Собираем параметры
        model = model or self.default_model
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        # Формируем сообщения
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Контент пользователя
        if image_input is not None:
            # Мультимодальный запрос
            img_base64 = self._encode_image(image_input)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        # Вызов API
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return response.choices[0].message.content#.strip()


# ========== Пример использования ==========
if __name__ == "__main__":
    # Убедитесь, что переменная OPENROUTER_API_KEY установлена
    # или передайте ключ явно: api_key="sk-..."

    llm = OpenRouterInference()

    # 1. Текстовый запрос (постобработка OCR)
    raw_ocr = "Это пример сырого текста с ошшибками"
    cleaned = llm.generate(
        user_prompt=f"Исправь ошибки OCR: {raw_ocr}",
        system_prompt="Ты — эксперт по постобработке OCR. Верни только исправленный текст.",
        temperature=0.1,
    )
    print("Очищенный текст:", cleaned)

    # 2. Суммаризация
    summary = llm.generate(
        user_prompt=cleaned,
        system_prompt="Сделай краткую выжимку (3-5 ключевых тезисов).",
        temperature=0.5,
    )
    print("Суммаризация:", summary)

    # 3. Мультимодальный запрос (если нужно распознать текст с картинки)
    # Для OCR лучше использовать специализированный OCR-движок, но если нужно,
    # можно попросить модель описать изображение:
    # description = llm.generate(
    #     user_prompt="Что изображено на картинке?",
    #     image_input="path/to/photo.jpg",
    # )
    # print("Описание:", description)