# -*- coding: utf-8 -*-
"""
Bộ não tìm kiếm dùng chung cho mọi nền tảng bot (Telegram, Messenger...).
Tách riêng ra đây để không phải viết/sửa 2 lần khi thêm nền tảng mới.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_PATH = BASE_DIR / "templates.json"
ALIASES_PATH = BASE_DIR / "aliases.json"

SUGGESTION_LIMIT = 8      # số gợi ý tối đa khi không chắc chắn
EXACT_MATCH_SCORE = 150   # ngưỡng để nhận biết đây là khớp chuỗi con/đầu chuỗi tuyệt đối (150-199)
AUTO_ANSWER_MARGIN = 15   # với điểm mờ (0-100), cần hơn kết quả #2 bấy nhiêu điểm mới tự trả lời luôn
MIN_SUGGEST_SCORE = 40    # dưới điểm này thì bỏ, không gợi ý (quá không liên quan, kiểu gõ linh tinh)
WORD_MATCH_THRESHOLD = 75  # 2 từ được coi là "khớp" khi độ giống nhau (0-100) >= ngưỡng này

# CHỈ những từ nối/giới từ thuần không mang nghĩa mới bị loại khi so khớp theo từ.
# Cố tình GIỮ LẠI các từ phủ định như "không", "chưa" và từ nghiệp vụ như "đơn", "hàng"
# vì chúng đổi hẳn ý nghĩa câu (VD "giao không thành công" ngược nghĩa với "giao thành công").
STOPWORDS = {
    "va", "hoac", "voi", "la", "the", "nay", "ve", "tren", "cua", "mot",
    "nhung", "den", "truoc", "nhu", "cho",
}


def normalize_no_accent(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.lower()


def content_words(norm_text: str):
    return [w for w in re.split(r"[^a-z0-9]+", norm_text) if len(w) >= 2 and w not in STOPWORDS]


def word_overlap_score(query_words, target_words):
    """
    Chấm điểm mức độ trùng khớp giữa 2 tập từ, chấp nhận sai chính tả từng từ
    (dùng fuzz.ratio ở cấp độ từ, không phải cấp độ cả câu).
    Trả về điểm 0-100 = (tỉ lệ % từ query tìm được) x (độ giống trung bình của các từ khớp).
    """
    if not query_words or not target_words:
        return 0.0
    total_quality = 0.0
    hits = 0
    for qw in query_words:
        best = max((fuzz.ratio(qw, tw) for tw in target_words), default=0)
        if best >= WORD_MATCH_THRESHOLD:
            hits += 1
            total_quality += best
    if hits == 0:
        return 0.0
    coverage = hits / len(query_words)
    quality = total_quality / hits
    return coverage * quality


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TemplateStore:
    """Giữ toàn bộ dữ liệu mẫu văn bản trong RAM, hỗ trợ tìm kiếm nhanh."""

    def __init__(self):
        self.templates = []
        self.by_slug = {}
        self.by_id = {}
        self.search_index = []
        self.title_words = []
        self.body_words = []
        self.alias_index = []  # [(alias_khong_dau, template), ...] dùng cho exact-match
        self.reload()

    def reload(self):
        templates = load_json(TEMPLATES_PATH, [])
        aliases = load_json(ALIASES_PATH, {})  # {"alias_command": id_or_slug}

        self.templates = templates
        self.by_slug = {t["slug"]: t for t in templates}
        self.by_id = {t["id"]: t for t in templates}
        self.search_index = [
            (normalize_no_accent(t["title"]), t) for t in templates
        ]
        self.title_words = [
            content_words(normalize_no_accent(t["title"])) for t in templates
        ]
        self.body_words = [
            content_words(normalize_no_accent(t["title"] + " " + t["text"]))
            for t in templates
        ]

        self.alias_index = []
        for alias, target in aliases.items():
            alias = alias.lstrip("/").strip().lower()
            template = None
            if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
                template = self.by_id.get(int(target))
            if template is None:
                template = self.by_slug.get(str(target))
            if template:
                self.by_slug[alias] = template
                self.alias_index.append((normalize_no_accent(alias), template))
            else:
                logger.warning("Alias '%s' -> '%s' khong khop mau nao", alias, target)

        logger.info(
            "Da nap %d mau van ban, %d alias.", len(self.templates), len(aliases)
        )

    def get(self, slug_or_alias: str):
        return self.by_slug.get(slug_or_alias.lstrip("/").strip().lower())

    def exact_match(self, query: str):
        """
        Ctrl+F thật sự: chỉ trả về mẫu khi từ khóa TRÙNG TUYỆT ĐỐI (sau khi bỏ dấu,
        viết thường) với một alias hoặc với tiêu đề mẫu. Không fuzzy, không đoán —
        có kết quả thì chắc chắn đúng, không có thì trả None để đi tiếp qua fuzzy.
        Nếu 2 mẫu khác nhau cùng khớp tuyệt đối (đụng alias) -> coi như không chắc
        chắn, trả None để không lỡ gửi nhầm.
        """
        q = normalize_no_accent(query.strip())
        if not q:
            return None

        hits = []
        for alias_norm, template in self.alias_index:
            if q == alias_norm:
                hits.append(template)
        for norm_title, template in self.search_index:
            if q == norm_title:
                hits.append(template)

        unique_ids = {t["id"] for t in hits}
        if len(unique_ids) == 1:
            return hits[0]
        return None  # rỗng hoặc đụng nhiều mẫu -> không tự trả lời, để fuzzy/suggest xử lý

    def search_scored(self, query: str, limit: int = SUGGESTION_LIMIT):
        """
        Tìm kiếm mờ: chấp nhận gõ sai chính tả từng từ, thiếu dấu, đảo thứ tự từ,
        hoặc từ khóa chỉ xuất hiện trong nội dung mẫu (không nằm trong tiêu đề).
        Trả về: list[(score, template)] sắp xếp giảm dần, không bao giờ rỗng một cách
        vô lý nếu có mẫu liên quan dù chỉ chút ít.
        """
        q = normalize_no_accent(query.strip())
        if not q or not self.search_index:
            return []

        q_words = content_words(q)
        results = []
        for idx, (norm_title, t) in enumerate(self.search_index):
            if norm_title.startswith(q):
                score = 199.0
            elif q in norm_title:
                score = 196.0
            else:
                title_score = word_overlap_score(q_words, self.title_words[idx])
                body_score = word_overlap_score(q_words, self.body_words[idx]) * 0.85
                whole_score = fuzz.WRatio(q, norm_title) * 0.55
                score = max(title_score, body_score, whole_score)
            results.append((score, t))

        results.sort(key=lambda x: x[0], reverse=True)
        results = [r for r in results if r[0] >= MIN_SUGGEST_SCORE][:limit]
        return results

    def search(self, query: str, limit: int = SUGGESTION_LIMIT):
        return [t for _, t in self.search_scored(query, limit=limit)]


def chunk_text(text: str, limit: int = 1900):
    """Chia văn bản dài thành nhiều đoạn để không vượt giới hạn ký tự của nền tảng chat."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts


def decide_answer(store: TemplateStore, query: str):
    """
    Logic quyết định dùng chung cho mọi nền tảng:
    - Trả về ("answer", template) nếu chắc chắn -> gửi thẳng nội dung.
    - Trả về ("suggest", [template,...]) nếu chưa chắc -> gợi ý danh sách.
    - Trả về ("empty", None) nếu không tìm thấy gì liên quan.
    """
    # Lớp 1: exact-match tuyệt đối theo alias/tiêu đề (kiểu Ctrl+F) — kiểm tra trước
    # tiên, không qua fuzzy. Đảm bảo gõ đúng alias (VD "PUD", "GTC") luôn ra đúng mẫu.
    exact = store.exact_match(query)
    if exact is not None:
        return "answer", exact

    scored = store.search_scored(query)
    if not scored:
        return "empty", None

    top_score, top_template = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0

    is_exact = top_score >= EXACT_MATCH_SCORE
    is_clear_fuzzy_winner = top_score >= 80 and (top_score - second_score) >= AUTO_ANSWER_MARGIN
    if is_exact or is_clear_fuzzy_winner:
        return "answer", top_template

    return "suggest", [t for _, t in scored]
