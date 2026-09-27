from dataclasses import dataclass
from typing import Optional
from difflib import SequenceMatcher
from urllib.parse import urljoin
import re

from config import SEARCH_SCORE_TRESHOLD

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://vino-svoe.ru"
WINES_URL = f"{BASE_URL}/wines"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0 Safari/537.36"
    )
}


# ============================================================
# Data
# ============================================================

@dataclass
class OCRData:
    wine_name: Optional[str] = None
    producer: Optional[str] = None
    grape_varieties: Optional[list[str]] = None
    color: Optional[str] = None
    category: Optional[str] = None
    region: Optional[str] = None
    vintage: Optional[int] = None
    alcohol: Optional[float] = None
    raw_text: str = ""


@dataclass
class Wine:
    url: str
    name: str
    producer: str
    grape_varieties: list[str]
    color: Optional[str]
    category: Optional[str]
    region: Optional[str]
    alcohol: Optional[float] = None
    vintage: Optional[int] = None


@dataclass
class SearchResult:
    wine: Wine
    score: float
    details: dict[str, float]


# ============================================================
# Searcher
# ============================================================

class WineSearcher:

    COLORS = (
        "белое",
        "красное",
        "оранжевое",
        "розовое",
    )

    CATEGORIES = (
        "экстра брют",
        "брют",
        "сухое",
        "полусухое",
        "полусладкое",
        "сладкое",
    )

    WEIGHTS = {
        "name": 0.35,
        "producer": 0.25,
        "grape": 0.15,
        "category": 0.10,
        "color": 0.07,
        "region": 0.08,
    }

    def __init__(
        self,
        timeout: int = 20,
        max_candidates: int = 30,
    ):
        self.timeout = timeout
        self.max_candidates = max_candidates

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    def _get_soup(self, url: str) -> BeautifulSoup:
        response = self.session.get(
            url,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return BeautifulSoup(
            response.text,
            "html.parser",
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize(text: Optional[str]) -> str:
        if not text:
            return ""

        text = text.lower()

        # Частые OCR-ошибки.
        # Можно добавить
        replacements = {
            "0": "о",
            "ё": "е"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[«»\"']", "", text)

        return text.strip()

    @classmethod
    def _similarity(
        cls,
        a: Optional[str],
        b: Optional[str],
    ) -> float:

        a = cls._normalize(a)
        b = cls._normalize(b)

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        if a in b or b in a:
            return 0.95

        return SequenceMatcher(None, a, b).ratio()

    # --------------------------------------------------------
    # Listing
    # --------------------------------------------------------

    def _get_candidate_urls(
        self,
        ocr: OCRData,
    ) -> list[str]:

        soup = self._get_soup(WINES_URL)

        candidates = []

        for link in soup.find_all("a", href=True):

            href = link["href"]

            if not href.startswith("/wines/"):
                continue

            # Сам /wines не является карточкой.
            if href == "/wines/":
                continue

            url = urljoin(BASE_URL, href)

            if url not in candidates:
                candidates.append(url)

        # Если удалось получить кандидатов с первой страницы,
        # предварительно отбираем их по тексту карточки.
        filtered = []

        search_values = [
            ocr.wine_name,
            ocr.producer,
            ocr.region,
        ]

        search_values = [
            self._normalize(value)
            for value in search_values
            if value
        ]

        for link in soup.find_all("a", href=True):

            href = link["href"]

            if not href.startswith("/wines/"):
                continue

            url = urljoin(BASE_URL, href)

            card_text = self._normalize(
                link.parent.get_text(" ", strip=True)
                if link.parent
                else link.get_text(" ", strip=True)
            )

            if not search_values:
                filtered.append(url)
                continue

            if any(
                value in card_text
                for value in search_values
            ):
                filtered.append(url)

        # Если предварительная фильтрация ничего не дала,
        # возвращаем все найденные карточки.
        if filtered:
            candidates = filtered

        # Убираем дубликаты, сохраняя порядок.
        result = []

        for url in candidates:
            if url not in result:
                result.append(url)

            if len(result) >= self.max_candidates:
                break

        return result

    # --------------------------------------------------------
    # Detail page
    # --------------------------------------------------------

    def _extract_detail(
        self,
        url: str,
    ) -> Optional[Wine]:

        soup = self._get_soup(url)

        text = soup.get_text(
            " ",
            strip=True,
        )

        # ----------------------------------------------------
        # Название
        # ----------------------------------------------------

        name = ""

        h1 = soup.find("h1")

        if h1:
            name = h1.get_text(
                " ",
                strip=True,
            )

        # ----------------------------------------------------
        # Производитель
        # ----------------------------------------------------

        producer = ""

        if h1:
            # Ищем первый текстовый элемент после h1,
            # который не является служебным блоком.
            for element in h1.find_all_next(
                string=True
            ):
                value = element.strip()

                if not value:
                    continue

                value = re.sub(
                    r"\s+",
                    " ",
                    value,
                )

                if value == name:
                    continue

                if len(value) > 2:
                    producer = value
                    break

        # Дополнительная попытка через title страницы.
        if not producer:
            title = soup.title

            if title:
                title_text = title.get_text(
                    " ",
                    strip=True,
                )

                if name and name in title_text:
                    producer = title_text.replace(
                        name,
                        "",
                    ).strip(" -|")

        # ----------------------------------------------------
        # Регион
        # ----------------------------------------------------

        region = self._extract_field(
            text,
            [
                "Регион",
            ],
            [
                "Сорт винограда",
                "Сорта винограда",
                "Категория и цвет",
            ],
        )

        # ----------------------------------------------------
        # Виноград
        # ----------------------------------------------------

        grape_text = self._extract_field(
            text,
            [
                "Сорт винограда",
                "Сорта винограда",
            ],
            [
                "Категория и цвет",
                "Температура подачи",
                "Крепость вина",
            ],
        )

        grape_varieties = self._parse_grapes(
            grape_text
        )

        # ----------------------------------------------------
        # Категория + цвет
        # ----------------------------------------------------

        category_color = self._extract_field(
            text,
            [
                "Категория и цвет",
            ],
            [
                "Температура подачи",
                "Крепость вина",
                "Сочетание с блюдами",
            ],
        )

        color = self._extract_color(
            category_color
        )

        category = self._extract_category(
            category_color
        )

        # ----------------------------------------------------
        # Крепость
        # ----------------------------------------------------

        alcohol = self._extract_alcohol(
            text
        )

        if not name:
            return None

        return Wine(
            url=url,
            name=name,
            producer=producer,
            grape_varieties=grape_varieties,
            color=color,
            category=category,
            region=region,
            alcohol=alcohol,
        )

    # --------------------------------------------------------
    # Field extraction
    # --------------------------------------------------------

    @classmethod
    def _extract_field(
        cls,
        text: str,
        labels: list[str],
        next_labels: list[str],
    ) -> str:

        normalized_text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        labels_pattern = "|".join(
            re.escape(label)
            for label in labels
        )

        next_pattern = "|".join(
            re.escape(label)
            for label in next_labels
        )

        pattern = (
            rf"(?:{labels_pattern})"
            rf"\s*"
            rf"(.*?)"
            rf"(?=(?:{next_pattern})|$)"
        )

        match = re.search(
            pattern,
            normalized_text,
            flags=re.IGNORECASE,
        )

        if not match:
            return ""

        return match.group(1).strip()

    @classmethod
    def _extract_color(
        cls,
        text: str,
    ) -> Optional[str]:

        normalized = cls._normalize(text)

        for color in cls.COLORS:
            if color in normalized:
                return color.capitalize()

        return None

    @classmethod
    def _extract_category(
        cls,
        text: str,
    ) -> Optional[str]:

        normalized = cls._normalize(text)

        # Важно: сначала проверяем более длинные значения.
        for category in cls.CATEGORIES:
            if category in normalized:
                return category.capitalize()

        return None

    @staticmethod
    def _parse_grapes(
        text: str,
    ) -> list[str]:

        if not text:
            return []

        # На сайте может быть несколько сортов.
        parts = re.split(
            r",|;|/|\n",
            text,
        )

        result = []

        for part in parts:
            part = part.strip()

            if not part:
                continue

            if part not in result:
                result.append(part)

        return result

    @staticmethod
    def _extract_alcohol(
        text: str,
    ) -> Optional[float]:

        match = re.search(
            r"Крепость\s*вина\s*(\d+(?:[.,]\d+)?)\s*%",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        return float(
            match.group(1).replace(",", ".")
        )

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    def _score(
        self,
        ocr: OCRData,
        wine: Wine,
    ) -> SearchResult:

        details = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        # Название
        if ocr.wine_name:
            score = self._similarity(
                ocr.wine_name,
                wine.name,
            )

            details["name"] = score

            weighted_sum += (
                score * self.WEIGHTS["name"]
            )

            weight_sum += self.WEIGHTS["name"]

        # Производитель
        if ocr.producer:
            score = self._similarity(
                ocr.producer,
                wine.producer,
            )

            details["producer"] = score

            weighted_sum += (
                score * self.WEIGHTS["producer"]
            )

            weight_sum += self.WEIGHTS["producer"]

        # Виноград
        if ocr.grape_varieties:
            grape_scores = []

            for query_grape in ocr.grape_varieties:

                scores = [
                    self._similarity(
                        query_grape,
                        wine_grape,
                    )
                    for wine_grape in wine.grape_varieties
                ]

                if scores:
                    grape_scores.append(
                        max(scores)
                    )

            score = (
                sum(grape_scores) / len(grape_scores)
                if grape_scores
                else 0.0
            )

            details["grape"] = score

            weighted_sum += (
                score * self.WEIGHTS["grape"]
            )

            weight_sum += self.WEIGHTS["grape"]

        # Категория
        if ocr.category:
            score = self._similarity(
                ocr.category,
                wine.category,
            )

            details["category"] = score

            weighted_sum += (
                score * self.WEIGHTS["category"]
            )

            weight_sum += self.WEIGHTS["category"]

        # Цвет
        if ocr.color:
            score = self._similarity(
                ocr.color,
                wine.color,
            )

            details["color"] = score

            weighted_sum += (
                score * self.WEIGHTS["color"]
            )

            weight_sum += self.WEIGHTS["color"]

        # Регион
        if ocr.region:
            score = self._similarity(
                ocr.region,
                wine.region,
            )

            details["region"] = score

            weighted_sum += (
                score * self.WEIGHTS["region"]
            )

            weight_sum += self.WEIGHTS["region"]

        if weight_sum == 0:
            total_score = 0.0
        else:
            total_score = weighted_sum / weight_sum

        # ----------------------------------------------------
        # Дополнительные комбинационные бонусы
        # ----------------------------------------------------

        if (
            details.get("producer", 0) >= 0.9
            and details.get("grape", 0) >= 0.9
        ):
            total_score += 0.05

        if (
            details.get("producer", 0) >= 0.9
            and details.get("name", 0) >= 0.9
        ):
            total_score += 0.05

        if (
            details.get("name", 0) >= 0.9
            and details.get("grape", 0) >= 0.9
        ):
            total_score += 0.05

        if (
            details.get("color", 0) >= 0.95
            and details.get("category", 0) >= 0.95
        ):
            total_score += 0.03

        available_scores = [
            score
            for score in details.values()
        ]

        if (
            len(available_scores) >= 3
            and all(score >= 0.9 for score in available_scores)
        ):
            total_score += 0.05

        total_score = min(
            total_score,
            1.0,
        )

        return SearchResult(
            wine=wine,
            score=total_score,
            details=details,
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def search(
        self,
        ocr: OCRData,
        top_k: int = 5,
    ) -> list[SearchResult]:

        candidate_urls = self._get_candidate_urls(
            ocr
        )

        results = []

        for url in candidate_urls:

            try:
                wine = self._extract_detail(url)

            except requests.RequestException:
                continue

            if wine is None:
                continue

            result = self._score(
                ocr,
                wine,
            )

            print(f"Score: {result.score}")
            if result.score >= SEARCH_SCORE_TRESHOLD:
                results.append(result)
                            
        results.sort(
            key=lambda x: x.score,
            reverse=True,
        )
        

        return results[:top_k]