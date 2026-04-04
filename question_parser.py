"""
question_parser.py -- Bilingual question parsing with graceful semantic fallback.

Priority order:
1. sentence-transformers semantic retrieval when available
2. scikit-learn TF-IDF similarity when available
3. keyword/rule matching as a final fallback
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Optional


class QuestionIntent(enum.Enum):
    WHY_THIS_ACTION = "why_this_action"
    WHY_NOT_OTHER = "why_not_other"
    MONSTER_INFLUENCE = "monster_influence"
    PATH_REASON = "path_reason"
    SAFETY_REASON = "safety_reason"
    GOAL_REASON = "goal_reason"
    DOT_COLLECTION = "dot_collection"
    GENERAL = "general"
    IRRELEVANT = "irrelevant"


@dataclass
class ParsedQuestion:
    original_text: str
    language: str
    intent: QuestionIntent
    confidence: float = 0.0
    mentioned_action: Optional[str] = None
    mentioned_monster_id: Optional[int] = None
    negation: bool = False
    keywords: list[str] = field(default_factory=list)


_INTENT_REFERENCES: dict[QuestionIntent, list[str]] = {
    QuestionIntent.WHY_THIS_ACTION: [
        "为什么选择这个动作？",
        "为什么这一步往左走？",
        "为什么现在向上移动？",
        "这一回合为什么这样走？",
        "Why did you choose this action?",
        "Why move this way?",
        "Why go left here?",
        "What is the reason for this step?",
    ],
    QuestionIntent.WHY_NOT_OTHER: [
        "为什么不往右走？",
        "为什么不选另一个方向？",
        "为什么不停下？",
        "Why not go right?",
        "Why not choose another direction?",
        "Why did you not stay?",
        "Why didn't you go left instead?",
    ],
    QuestionIntent.MONSTER_INFLUENCE: [
        "怪物影响了这次决策吗？",
        "怪物#3为什么危险？",
        "你是在躲怪物吗？",
        "How did the monster affect the decision?",
        "Is monster #2 dangerous?",
        "Are you avoiding a monster?",
    ],
    QuestionIntent.PATH_REASON: [
        "为什么走这条路线？",
        "为什么选这条路？",
        "路径规划依据是什么？",
        "Why this path?",
        "Why take this route?",
        "What is the route strategy?",
    ],
    QuestionIntent.SAFETY_REASON: [
        "这里安全吗？",
        "现在危险吗？",
        "碰撞风险高吗？",
        "Is it safe here?",
        "Am I in danger?",
        "What is the collision risk?",
    ],
    QuestionIntent.GOAL_REASON: [
        "离出口还有多远？",
        "我们在接近终点吗？",
        "出口什么时候打开？",
        "How far is the exit?",
        "Are we making progress toward the goal?",
        "When does the exit open?",
    ],
    QuestionIntent.DOT_COLLECTION: [
        "为什么去吃那个豆子？",
        "还剩多少豆子？",
        "最近的豆子在哪？",
        "Why collect that dot?",
        "How many dots are left?",
        "Where is the nearest pellet?",
        "What is the dot collection strategy?",
    ],
    QuestionIntent.GENERAL: [
        "现在情况怎么样？",
        "当前局面是什么？",
        "发生了什么？",
        "What is happening now?",
        "What is the current state?",
        "Summarize the situation.",
    ],
}

_IRRELEVANT_REFERENCES = [
    "今天天气怎么样？",
    "你叫什么名字？",
    "给我讲个笑话。",
    "推荐一部电影。",
    "What time is it?",
    "Tell me a joke.",
    "What is your name?",
    "Recommend a movie.",
]

_ACTION_KW_EN = {
    "up": "UP",
    "north": "UP",
    "down": "DOWN",
    "south": "DOWN",
    "left": "LEFT",
    "west": "LEFT",
    "right": "RIGHT",
    "east": "RIGHT",
    "stay": "STAY",
    "wait": "STAY",
    "stop": "STAY",
}

_ACTION_KW_ZH = {
    "向上": "UP",
    "往上": "UP",
    "上走": "UP",
    "向下": "DOWN",
    "往下": "DOWN",
    "下走": "DOWN",
    "左转": "LEFT",
    "向左": "LEFT",
    "往左": "LEFT",
    "右转": "RIGHT",
    "向右": "RIGHT",
    "往右": "RIGHT",
    "不动": "STAY",
    "原地": "STAY",
    "停下": "STAY",
    "上": "UP",
    "下": "DOWN",
    "左": "LEFT",
    "右": "RIGHT",
}

_NEGATION_EN = [
    "why not",
    "why didn't",
    "instead of",
    "rather than",
    "not go",
    "not move",
]

_NEGATION_ZH = [
    "为什么不",
    "为何不",
    "不去",
    "而不是",
    "不选",
    "不走",
    "不往",
]

_RULE_HINTS = {
    QuestionIntent.WHY_THIS_ACTION: {
        "en": ["why", "reason", "step", "move"],
        "zh": ["为什么", "原因", "这一步", "动作", "决策"],
    },
    QuestionIntent.WHY_NOT_OTHER: {
        "en": ["why not", "instead", "rather than"],
        "zh": ["为什么不", "而不是", "不选", "不走"],
    },
    QuestionIntent.MONSTER_INFLUENCE: {
        "en": ["monster", "ghost", "dangerous", "avoid"],
        "zh": ["怪物", "鬼", "危险", "躲", "避开"],
    },
    QuestionIntent.PATH_REASON: {
        "en": ["path", "route", "corridor", "way"],
        "zh": ["路径", "路线", "路", "走法"],
    },
    QuestionIntent.SAFETY_REASON: {
        "en": ["safe", "danger", "risk", "collision"],
        "zh": ["安全", "危险", "风险", "碰撞"],
    },
    QuestionIntent.GOAL_REASON: {
        "en": ["exit", "goal", "finish", "progress"],
        "zh": ["出口", "终点", "目标", "进度", "打开"],
    },
    QuestionIntent.DOT_COLLECTION: {
        "en": ["dot", "dots", "pellet", "pellets", "collect"],
        "zh": ["豆", "豆子", "吃豆", "收集"],
    },
    QuestionIntent.GENERAL: {
        "en": ["state", "situation", "summary", "what is happening"],
        "zh": ["情况", "状态", "发生", "总结"],
    },
    QuestionIntent.IRRELEVANT: {
        "en": ["weather", "time", "joke", "movie", "name"],
        "zh": ["天气", "时间", "笑话", "电影", "名字"],
    },
}

_GAME_CONTEXT_TERMS_EN = {
    "monster",
    "ghost",
    "exit",
    "goal",
    "path",
    "route",
    "safe",
    "danger",
    "risk",
    "dot",
    "pellet",
    "maze",
    "pacman",
    "pac-man",
    "move",
    "left",
    "right",
    "up",
    "down",
    "stay",
}

_GAME_CONTEXT_TERMS_ZH = {
    "怪物",
    "出口",
    "目标",
    "路径",
    "路线",
    "安全",
    "危险",
    "风险",
    "豆",
    "豆子",
    "迷宫",
    "吃豆人",
    "向左",
    "向右",
    "向上",
    "向下",
    "左",
    "右",
    "上",
    "下",
    "停下",
}


class SemanticMatcher:
    """Semantic classifier with layered fallbacks."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.backend = "rules"
        self._st_model = None
        self._tfidf_vectorizer = None
        self._reference_matrix = None

        self._reference_labels: list[QuestionIntent] = []
        self._reference_texts: list[str] = []
        for intent, refs in _INTENT_REFERENCES.items():
            for text in refs:
                self._reference_labels.append(intent)
                self._reference_texts.append(text)
        for text in _IRRELEVANT_REFERENCES:
            self._reference_labels.append(QuestionIntent.IRRELEVANT)
            self._reference_texts.append(text)

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(self.model_name)
            self._reference_matrix = self._st_model.encode(
                self._reference_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            self.backend = "sentence-transformers"
            return
        except Exception:
            self._st_model = None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self._reference_matrix = self._tfidf_vectorizer.fit_transform(self._reference_texts)
            self.backend = "tfidf"
        except Exception:
            self._tfidf_vectorizer = None
            self._reference_matrix = None
            self.backend = "rules"

    def classify(self, question: str, language: Optional[str] = None) -> tuple[QuestionIntent, float]:
        lang = language or QuestionParser.detect_language(question)
        semantic_scores = self._semantic_scores(question)
        rule_scores = self._rule_scores(question, lang)
        combined = self._combine_scores(semantic_scores, rule_scores)

        best_intent = max(combined, key=combined.get)
        best_score = combined[best_intent]

        irrelevant_score = combined.get(QuestionIntent.IRRELEVANT, 0.0)
        game_scores = {
            intent: score
            for intent, score in combined.items()
            if intent != QuestionIntent.IRRELEVANT
        }
        best_game_intent = max(game_scores, key=game_scores.get)
        best_game_score = game_scores[best_game_intent]

        if not self._looks_game_related(question, lang) and best_game_score < 0.42:
            return QuestionIntent.IRRELEVANT, round(max(irrelevant_score, 0.75), 3)

        if irrelevant_score >= max(best_game_score + 0.08, 0.5):
            return QuestionIntent.IRRELEVANT, round(irrelevant_score, 3)

        if best_game_score < 0.28:
            if self._looks_game_related(question, lang):
                return QuestionIntent.GENERAL, round(max(best_game_score, 0.35), 3)
            return QuestionIntent.IRRELEVANT, round(max(irrelevant_score, 0.65), 3)

        return best_game_intent, round(best_game_score, 3)

    def _semantic_scores(self, question: str) -> dict[QuestionIntent, float]:
        scores_by_intent = {intent: 0.0 for intent in QuestionIntent}

        if self.backend == "sentence-transformers" and self._st_model is not None:
            query_vec = self._st_model.encode(
                question,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            similarities = (query_vec @ self._reference_matrix.T).detach().cpu().tolist()
            return self._aggregate_similarity_scores(similarities)

        if self.backend == "tfidf" and self._tfidf_vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self._tfidf_vectorizer.transform([question])
            similarities = cosine_similarity(query_vec, self._reference_matrix)[0].tolist()
            return self._aggregate_similarity_scores(similarities)

        return scores_by_intent

    def _aggregate_similarity_scores(self, scores: list[float]) -> dict[QuestionIntent, float]:
        grouped: dict[QuestionIntent, list[float]] = {intent: [] for intent in QuestionIntent}
        for label, score in zip(self._reference_labels, scores):
            grouped[label].append(float(score))

        aggregated: dict[QuestionIntent, float] = {}
        for intent, values in grouped.items():
            if not values:
                aggregated[intent] = 0.0
                continue
            top_values = sorted(values, reverse=True)[:2]
            aggregated[intent] = sum(top_values) / len(top_values)
        return aggregated

    def _rule_scores(self, question: str, lang: str) -> dict[QuestionIntent, float]:
        haystack = question.lower() if lang == "en" else question
        scores = {intent: 0.0 for intent in QuestionIntent}

        for intent, hints in _RULE_HINTS.items():
            matched = [term for term in hints[lang] if term in haystack]
            if matched:
                scores[intent] = max(scores[intent], min(0.92, 0.38 + 0.17 * len(matched)))

        action = QuestionParser.extract_action(question, lang)
        monster_id = QuestionParser.extract_monster_id(question)
        negation = QuestionParser.detect_negation(question.lower(), lang)

        if negation and action:
            scores[QuestionIntent.WHY_NOT_OTHER] = max(scores[QuestionIntent.WHY_NOT_OTHER], 0.9)
        elif action and ("why" in question.lower() or "为什么" in question):
            scores[QuestionIntent.WHY_THIS_ACTION] = max(scores[QuestionIntent.WHY_THIS_ACTION], 0.72)

        if monster_id is not None:
            scores[QuestionIntent.MONSTER_INFLUENCE] = max(scores[QuestionIntent.MONSTER_INFLUENCE], 0.88)

        if self._looks_game_related(question, lang):
            scores[QuestionIntent.GENERAL] = max(scores[QuestionIntent.GENERAL], 0.35)
        else:
            scores[QuestionIntent.IRRELEVANT] = max(scores[QuestionIntent.IRRELEVANT], 0.7)

        return scores

    def _combine_scores(
        self,
        semantic_scores: dict[QuestionIntent, float],
        rule_scores: dict[QuestionIntent, float],
    ) -> dict[QuestionIntent, float]:
        if self.backend == "sentence-transformers":
            semantic_weight, rule_weight = 0.78, 0.22
        elif self.backend == "tfidf":
            semantic_weight, rule_weight = 0.62, 0.38
        else:
            semantic_weight, rule_weight = 0.0, 1.0

        combined = {}
        for intent in QuestionIntent:
            combined[intent] = (
                semantic_scores.get(intent, 0.0) * semantic_weight
                + rule_scores.get(intent, 0.0) * rule_weight
            )
        return combined

    @staticmethod
    def _looks_game_related(question: str, lang: str) -> bool:
        haystack = question.lower() if lang == "en" else question
        terms = _GAME_CONTEXT_TERMS_EN if lang == "en" else _GAME_CONTEXT_TERMS_ZH
        return any(term in haystack for term in terms)


class QuestionParser:
    """Parse a free-form bilingual question into a structured intent."""

    def __init__(self, semantic: bool = True):
        self.semantic_enabled = semantic
        self.semantic_matcher = SemanticMatcher() if semantic else None
        self.backend = self.semantic_matcher.backend if self.semantic_matcher else "rules"

    def parse(self, text: str) -> ParsedQuestion:
        text = text.strip()
        if not text:
            return ParsedQuestion(
                original_text=text,
                language="en",
                intent=QuestionIntent.GENERAL,
                confidence=0.0,
            )

        language = self.detect_language(text)
        action = self.extract_action(text, language)
        monster_id = self.extract_monster_id(text)
        negation = self.detect_negation(text.lower(), language)
        keywords = self.extract_keywords(text, language)

        if self.semantic_matcher is not None:
            intent, confidence = self.semantic_matcher.classify(text, language)
        else:
            matcher = SemanticMatcher()
            matcher.backend = "rules"
            intent, confidence = matcher.classify(text, language)

        if negation and action and intent != QuestionIntent.IRRELEVANT:
            intent = QuestionIntent.WHY_NOT_OTHER
            confidence = max(confidence, 0.8)
        elif monster_id is not None and intent not in (QuestionIntent.IRRELEVANT, QuestionIntent.WHY_NOT_OTHER):
            intent = QuestionIntent.MONSTER_INFLUENCE
            confidence = max(confidence, 0.8)

        if intent == QuestionIntent.GENERAL and not keywords and not self._looks_game_related(text, language):
            intent = QuestionIntent.IRRELEVANT
            confidence = max(confidence, 0.65)

        return ParsedQuestion(
            original_text=text,
            language=language,
            intent=intent,
            confidence=round(float(confidence), 3),
            mentioned_action=action,
            mentioned_monster_id=monster_id,
            negation=negation,
            keywords=keywords,
        )

    @staticmethod
    def detect_language(text: str) -> str:
        return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en"

    @staticmethod
    def extract_action(text: str, lang: str) -> Optional[str]:
        primary = _ACTION_KW_ZH if lang == "zh" else _ACTION_KW_EN
        secondary = _ACTION_KW_EN if lang == "zh" else _ACTION_KW_ZH
        haystack = text if lang == "zh" else text.lower()
        alt_haystack = text.lower() if lang == "zh" else text

        for keyword in sorted(primary, key=len, reverse=True):
            if keyword in haystack:
                return primary[keyword]

        for keyword in sorted(secondary, key=len, reverse=True):
            if keyword in alt_haystack:
                return secondary[keyword]

        return None

    @staticmethod
    def extract_monster_id(text: str) -> Optional[int]:
        patterns = [
            r"(?:monster|ghost)\s*#?\s*(\d+)",
            r"怪物\s*#?\s*(\d+)",
            r"#\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def detect_negation(lower_text: str, lang: str) -> bool:
        patterns = _NEGATION_ZH if lang == "zh" else _NEGATION_EN
        return any(pattern in lower_text for pattern in patterns)

    @staticmethod
    def extract_keywords(text: str, lang: str) -> list[str]:
        haystack = text.lower() if lang == "en" else text
        keywords: list[str] = []

        hint_table = _RULE_HINTS
        for hints in hint_table.values():
            for keyword in hints[lang]:
                if keyword in haystack and keyword not in keywords:
                    keywords.append(keyword)

        action = QuestionParser.extract_action(text, lang)
        if action and action not in keywords:
            keywords.append(action)

        monster_id = QuestionParser.extract_monster_id(text)
        if monster_id is not None:
            keywords.append(f"monster#{monster_id}")

        return keywords

    @staticmethod
    def _looks_game_related(text: str, lang: str) -> bool:
        return SemanticMatcher._looks_game_related(text, lang)
