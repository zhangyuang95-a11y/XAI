"""
question_parser.py -- Bilingual question parsing with semantic fallback.

Priority order:
1. sentence-transformers semantic retrieval when available
2. scikit-learn TF-IDF similarity when available
3. keyword and rule matching as a final fallback
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
    POLICY_SUMMARY = "policy_summary"
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
    semantic_frame: dict[str, object] = field(default_factory=dict)
    grounded: bool = False
    relevance_reason: str = ""


_INTENT_REFERENCES: dict[QuestionIntent, list[str]] = {
    QuestionIntent.WHY_THIS_ACTION: [
        "为什么这样走？",
        "为什么向右走？",
        "为什么现在往上走？",
        "Why did you choose this action?",
        "Why move this way?",
        "Why go right here?",
    ],
    QuestionIntent.WHY_NOT_OTHER: [
        "为什么不往右走？",
        "为什么不选别的动作？",
        "为什么不向下？",
        "Why not go right?",
        "Why not choose another direction?",
        "Why didn't you go down instead?",
    ],
    QuestionIntent.MONSTER_INFLUENCE: [
        "怪物影响了这次决策吗？",
        "怪物#1为什么危险？",
        "你是在躲怪物吗？",
        "How did the monster affect the decision?",
        "Is monster #1 dangerous?",
        "Are you avoiding a monster?",
    ],
    QuestionIntent.PATH_REASON: [
        "为什么走这条路？",
        "为什么选这条路线？",
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
        "Where is the nearest dot?",
    ],
    QuestionIntent.POLICY_SUMMARY: [
        "你的整体策略是什么？",
        "你通常怎么决策？",
        "总结一下你的策略。",
        "What is your overall policy?",
        "How do you usually decide?",
        "Summarize your policy.",
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
    "给我讲个笑话。",
    "What time is it?",
    "Tell me a joke.",
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
    "上方": "UP",
    "向下": "DOWN",
    "往下": "DOWN",
    "下走": "DOWN",
    "下方": "DOWN",
    "左转": "LEFT",
    "向左": "LEFT",
    "往左": "LEFT",
    "左边": "LEFT",
    "右转": "RIGHT",
    "向右": "RIGHT",
    "往右": "RIGHT",
    "右边": "RIGHT",
    "不动": "STAY",
    "原地": "STAY",
    "停下": "STAY",
}

_NEGATION_EN = ["why not", "why didn't", "instead of", "rather than", "not go", "not move"]
_NEGATION_ZH = ["为什么不", "为何不", "不去", "而不是", "不选", "不走", "不往"]

_RULE_HINTS = {
    QuestionIntent.WHY_THIS_ACTION: {"en": ["why", "reason", "move", "step"], "zh": ["为什么", "原因", "动作", "决策"]},
    QuestionIntent.WHY_NOT_OTHER: {"en": ["why not", "instead", "rather than"], "zh": ["为什么不", "而不是", "不选", "不走"]},
    QuestionIntent.MONSTER_INFLUENCE: {"en": ["monster", "ghost", "danger", "avoid"], "zh": ["怪物", "危险", "躲", "避开"]},
    QuestionIntent.PATH_REASON: {"en": ["path", "route", "corridor"], "zh": ["路径", "路线", "走法"]},
    QuestionIntent.SAFETY_REASON: {"en": ["safe", "danger", "risk", "collision"], "zh": ["安全", "危险", "风险", "碰撞"]},
    QuestionIntent.GOAL_REASON: {"en": ["exit", "goal", "finish", "progress"], "zh": ["出口", "终点", "目标", "进度"]},
    QuestionIntent.DOT_COLLECTION: {"en": ["dot", "pellet", "collect"], "zh": ["豆", "豆子", "吃豆"]},
    QuestionIntent.POLICY_SUMMARY: {"en": ["policy", "overall", "usually", "summarize"], "zh": ["策略", "整体", "总体", "通常", "总结"]},
    QuestionIntent.GENERAL: {"en": ["state", "situation", "summary"], "zh": ["情况", "局面", "总结"]},
    QuestionIntent.IRRELEVANT: {"en": ["weather", "time", "joke", "movie"], "zh": ["天气", "时间", "笑话", "电影"]},
}

_GAME_CONTEXT_TERMS_EN = {
    "monster", "ghost", "exit", "goal", "path", "route", "safe", "danger", "risk",
    "dot", "pellet", "maze", "pacman", "pac-man", "move", "left", "right", "up", "down", "policy", "strategy",
}

_GAME_CONTEXT_TERMS_ZH = {
    "怪物", "出口", "目标", "路径", "路线", "安全", "危险", "风险",
    "豆", "豆子", "迷宫", "吃豆人", "向左", "向右", "向上", "向下", "策略", "决策", "动作",
}

_POLICY_SUMMARY_CUES_EN = (
    "overall policy",
    "policy summary",
    "summarize your policy",
    "summarize the policy",
    "how do you usually decide",
    "how do you decide",
    "how do you choose moves",
    "decision policy",
    "usual strategy",
)

_POLICY_SUMMARY_CUES_ZH = (
    "整体策略",
    "总体策略",
    "总结一下你的策略",
    "总结你的策略",
    "你通常怎么决策",
    "通常怎么决策",
    "如何决策",
    "怎么决策",
    "怎么做决定",
    "一般怎么选动作",
    "你是怎么做决定的",
)

_DOMAIN_CUES_EN = {
    "pacman", "pac-man", "maze", "game", "move", "moving", "go", "step",
    "action", "decision", "choose", "choice", "left", "right", "up", "down",
    "stay", "wait", "monster", "ghost", "dot", "pellet", "exit", "safe",
    "danger", "risk", "path", "route", "policy", "strategy",
}

_DOMAIN_CUES_ZH = {
    "\u5403\u8c46\u4eba",  # Pac-Man
    "\u8ff7\u5bab",
    "\u6e38\u620f",
    "\u8d70",
    "\u79fb\u52a8",
    "\u52a8\u4f5c",
    "\u51b3\u7b56",
    "\u9009\u62e9",
    "\u5411\u4e0a",
    "\u5411\u4e0b",
    "\u5411\u5de6",
    "\u5411\u53f3",
    "\u5f80\u4e0a",
    "\u5f80\u4e0b",
    "\u5f80\u5de6",
    "\u5f80\u53f3",
    "\u4e0a\u8d70",
    "\u4e0b\u8d70",
    "\u5de6\u8d70",
    "\u53f3\u8d70",
    "\u602a\u7269",
    "\u5e7d\u7075",
    "\u8c46\u5b50",
    "\u5403\u8c46",
    "\u51fa\u53e3",
    "\u5b89\u5168",
    "\u5371\u9669",
    "\u98ce\u9669",
    "\u8def\u7ebf",
    "\u8def\u5f84",
    "\u7b56\u7565",
    "\u8fd9\u4e00\u6b65",
    "\u8fd9\u6b65",
    "\u73b0\u5728",
    "\u5f53\u524d",
}

_CLEAR_OFF_TOPIC_CUES_EN = {
    "homework", "assignment", "essay", "write code", "write a poem",
    "weather", "movie", "joke", "recipe", "news",
}

_CLEAR_OFF_TOPIC_CUES_ZH = {
    "\u5199\u4f5c\u4e1a",
    "\u4f5c\u4e1a",
    "\u5199\u8bba\u6587",
    "\u5199\u4ee3\u7801",
    "\u5199\u8bd7",
    "\u5929\u6c14",
    "\u7535\u5f71",
    "\u7b11\u8bdd",
    "\u83dc\u8c31",
    "\u65b0\u95fb",
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
        return self._resolve_intent(question, lang, combined)

    @classmethod
    def classify_rules_only(cls, question: str, language: Optional[str] = None) -> tuple[QuestionIntent, float]:
        lang = language or QuestionParser.detect_language(question)
        matcher = cls.__new__(cls)
        matcher.backend = "rules"
        scores = matcher._rule_scores(question, lang)
        return cls._resolve_intent(question, lang, scores)

    @staticmethod
    def _resolve_intent(
        question: str,
        lang: str,
        combined: dict[QuestionIntent, float],
    ) -> tuple[QuestionIntent, float]:
        irrelevant_score = combined.get(QuestionIntent.IRRELEVANT, 0.0)
        game_scores = {intent: score for intent, score in combined.items() if intent != QuestionIntent.IRRELEVANT}
        best_game_intent = max(game_scores, key=game_scores.get)
        best_game_score = game_scores[best_game_intent]

        if not SemanticMatcher._looks_game_related(question, lang) and best_game_score < 0.42:
            return QuestionIntent.IRRELEVANT, round(max(irrelevant_score, 0.75), 3)
        if irrelevant_score >= max(best_game_score + 0.08, 0.5):
            return QuestionIntent.IRRELEVANT, round(irrelevant_score, 3)
        if best_game_score < 0.28:
            if SemanticMatcher._looks_game_related(question, lang):
                return QuestionIntent.GENERAL, round(max(best_game_score, 0.35), 3)
            return QuestionIntent.IRRELEVANT, round(max(irrelevant_score, 0.65), 3)
        return best_game_intent, round(best_game_score, 3)

    def _semantic_scores(self, question: str) -> dict[QuestionIntent, float]:
        scores = {intent: 0.0 for intent in QuestionIntent}
        if self.backend == "sentence-transformers" and self._st_model is not None:
            query_vec = self._st_model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
            similarities = (query_vec @ self._reference_matrix.T).detach().cpu().tolist()
            return self._aggregate_similarity_scores(similarities)
        if self.backend == "tfidf" and self._tfidf_vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self._tfidf_vectorizer.transform([question])
            similarities = cosine_similarity(query_vec, self._reference_matrix)[0].tolist()
            return self._aggregate_similarity_scores(similarities)
        return scores

    def _aggregate_similarity_scores(self, scores: list[float]) -> dict[QuestionIntent, float]:
        grouped: dict[QuestionIntent, list[float]] = {intent: [] for intent in QuestionIntent}
        for label, score in zip(self._reference_labels, scores):
            grouped[label].append(float(score))
        aggregated: dict[QuestionIntent, float] = {}
        for intent, values in grouped.items():
            top_values = sorted(values, reverse=True)[:2]
            aggregated[intent] = sum(top_values) / len(top_values) if top_values else 0.0
        return aggregated

    def _rule_scores(self, question: str, lang: str) -> dict[QuestionIntent, float]:
        haystack = question.lower() if lang == "en" else question
        scores = {intent: 0.0 for intent in QuestionIntent}

        for intent, hints in _RULE_HINTS.items():
            matched = [term for term in hints[lang] if term in haystack]
            if matched:
                scores[intent] = max(scores[intent], min(0.92, 0.38 + 0.17 * len(matched)))

        if QuestionParser._looks_like_policy_summary(question, lang):
            scores[QuestionIntent.POLICY_SUMMARY] = max(scores[QuestionIntent.POLICY_SUMMARY], 0.9)

        action = QuestionParser.extract_action(question, lang)
        monster_id = QuestionParser.extract_monster_id(question)
        negation = QuestionParser.detect_negation(question.lower() if lang == "en" else question, lang)

        if negation and action:
            scores[QuestionIntent.WHY_NOT_OTHER] = max(scores[QuestionIntent.WHY_NOT_OTHER], 0.9)
        elif action and (("why" in question.lower()) or ("为什么" in question)):
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
        return {
            intent: semantic_scores.get(intent, 0.0) * semantic_weight + rule_scores.get(intent, 0.0) * rule_weight
            for intent in QuestionIntent
        }

    @staticmethod
    def _looks_game_related(question: str, lang: str) -> bool:
        haystack = question.lower() if lang == "en" else question
        terms = _GAME_CONTEXT_TERMS_EN if lang == "en" else _GAME_CONTEXT_TERMS_ZH
        domain_terms = _DOMAIN_CUES_EN if lang == "en" else _DOMAIN_CUES_ZH
        return (
            QuestionParser._looks_like_policy_summary(question, lang)
            or any(term in haystack for term in terms)
            or any(term in haystack for term in domain_terms)
            or QuestionParser.extract_action(question, lang) is not None
            or QuestionParser.extract_monster_id(question) is not None
        )

    @staticmethod
    def _is_clearly_off_topic(question: str, lang: str) -> bool:
        haystack = question.lower() if lang == "en" else question
        cues = _CLEAR_OFF_TOPIC_CUES_EN if lang == "en" else _CLEAR_OFF_TOPIC_CUES_ZH
        return any(cue in haystack for cue in cues)


class QuestionParser:
    """Parse a free-form bilingual question into a structured intent."""

    def __init__(self, semantic: bool = True):
        self.semantic_matcher = SemanticMatcher() if semantic else None
        self.backend = self.semantic_matcher.backend if self.semantic_matcher else "rules"

    def parse(self, text: str) -> ParsedQuestion:
        text = text.strip()
        if not text:
            return ParsedQuestion(original_text=text, language="en", intent=QuestionIntent.GENERAL, confidence=0.0)

        language = self.detect_language(text)
        action = self.extract_action(text, language)
        monster_id = self.extract_monster_id(text)
        negation = self.detect_negation(text.lower() if language == "en" else text, language)
        keywords = self.extract_keywords(text, language)
        semantic_frame = self.build_semantic_frame(
            text=text,
            lang=language,
            action=action,
            monster_id=monster_id,
            negation=negation,
            keywords=keywords,
        )

        if not bool(semantic_frame["grounded"]):
            return ParsedQuestion(
                original_text=text,
                language=language,
                intent=QuestionIntent.IRRELEVANT,
                confidence=float(semantic_frame["confidence"]),
                mentioned_action=action,
                mentioned_monster_id=monster_id,
                negation=negation,
                keywords=keywords,
                semantic_frame=semantic_frame,
                grounded=False,
                relevance_reason=str(semantic_frame["reason"]),
            )

        frame_intent = semantic_frame.get("intent")
        if isinstance(frame_intent, QuestionIntent):
            intent = frame_intent
            confidence = float(semantic_frame["confidence"])
        elif self.semantic_matcher is not None:
            intent, confidence = self.semantic_matcher.classify(text, language)
        else:
            intent, confidence = SemanticMatcher.classify_rules_only(text, language)

        if self._looks_like_policy_summary(text, language):
            intent = QuestionIntent.POLICY_SUMMARY
            confidence = max(confidence, 0.8)
        elif negation and action and intent != QuestionIntent.IRRELEVANT:
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
            semantic_frame=semantic_frame,
            grounded=bool(semantic_frame["grounded"]),
            relevance_reason=str(semantic_frame["reason"]),
        )

    @staticmethod
    def build_semantic_frame(
        *,
        text: str,
        lang: str,
        action: Optional[str],
        monster_id: Optional[int],
        negation: bool,
        keywords: list[str],
    ) -> dict[str, object]:
        """Ground the question before intent classification.

        This is the source of truth for understanding: a question must refer to
        the game, a game entity, or the current decision before it can ask for a
        Pac-Man explanation. Semantic similarity is allowed only after this
        grounding step, so generic "why" questions do not become action
        explanations by accident.
        """

        haystack = text.lower() if lang == "en" else text
        off_topic_cues = _CLEAR_OFF_TOPIC_CUES_EN if lang == "en" else _CLEAR_OFF_TOPIC_CUES_ZH
        domain_cues = _DOMAIN_CUES_EN if lang == "en" else _DOMAIN_CUES_ZH
        legacy_domain_cues = _GAME_CONTEXT_TERMS_EN if lang == "en" else _GAME_CONTEXT_TERMS_ZH
        policy_cues = _POLICY_SUMMARY_CUES_EN if lang == "en" else _POLICY_SUMMARY_CUES_ZH

        matched_off_topic = [cue for cue in off_topic_cues if cue in haystack]
        if matched_off_topic:
            return {
                "domain": "off_topic",
                "topic": "off_topic",
                "speech_act": "why" if QuestionParser._contains_why(text, lang) else "other",
                "action": action,
                "monster_id": monster_id,
                "negation": negation,
                "keywords": keywords,
                "grounded": False,
                "intent": QuestionIntent.IRRELEVANT,
                "confidence": 0.95,
                "reason": f"off-topic cue: {matched_off_topic[0]}",
            }

        matched_domain = [cue for cue in domain_cues if cue in haystack]
        matched_legacy_domain = [cue for cue in legacy_domain_cues if cue in haystack]
        speech_act = QuestionParser._speech_act(text, lang)
        topic = QuestionParser._semantic_topic(text, lang, action, monster_id, matched_domain)
        grounded = bool(
            action
            or monster_id is not None
            or matched_domain
            or matched_legacy_domain
            or QuestionParser._looks_like_policy_summary(text, lang)
        )

        if not grounded:
            return {
                "domain": "unknown",
                "topic": "unknown",
                "speech_act": speech_act,
                "action": action,
                "monster_id": monster_id,
                "negation": negation,
                "keywords": keywords,
                "grounded": False,
                "intent": QuestionIntent.IRRELEVANT,
                "confidence": 0.9,
                "reason": "no Pac-Man entity, action, state, or policy concept was grounded",
            }

        intent = QuestionParser._intent_from_frame(
            topic=topic,
            speech_act=speech_act,
            action=action,
            monster_id=monster_id,
            negation=negation,
            policy_match=any(cue in haystack for cue in policy_cues),
        )
        confidence = 0.88 if action or monster_id is not None else 0.78
        if topic in {"safety", "policy", "goal", "dot", "path"}:
            confidence = max(confidence, 0.82)

        return {
            "domain": "pacman",
            "topic": topic,
            "speech_act": speech_act,
            "action": action,
            "monster_id": monster_id,
            "negation": negation,
            "keywords": keywords,
            "grounded": True,
            "intent": intent,
            "confidence": confidence,
            "reason": "grounded to Pac-Man state/action/entity",
        }

    @staticmethod
    def _contains_why(text: str, lang: str) -> bool:
        haystack = text.lower() if lang == "en" else text
        return ("why" in haystack) if lang == "en" else ("\u4e3a\u4ec0\u4e48" in haystack or "\u4e3a\u4f55" in haystack)

    @staticmethod
    def _speech_act(text: str, lang: str) -> str:
        haystack = text.lower() if lang == "en" else text
        if QuestionParser._contains_why(text, lang):
            return "why"
        if ("safe" in haystack or "danger" in haystack) if lang == "en" else ("\u5b89\u5168" in haystack or "\u5371\u9669" in haystack):
            return "safety_check"
        if ("summary" in haystack or "overall" in haystack) if lang == "en" else ("\u603b\u7ed3" in haystack or "\u6574\u4f53" in haystack):
            return "summary"
        return "ask"

    @staticmethod
    def _semantic_topic(
        text: str,
        lang: str,
        action: Optional[str],
        monster_id: Optional[int],
        matched_domain: list[str],
    ) -> str:
        haystack = text.lower() if lang == "en" else text
        if QuestionParser._looks_like_policy_summary(text, lang):
            return "policy"
        if monster_id is not None or ("monster" in haystack or "ghost" in haystack) or ("\u602a\u7269" in haystack or "\u5e7d\u7075" in haystack):
            return "monster"
        if ("safe" in haystack or "risk" in haystack or "danger" in haystack) or ("\u5b89\u5168" in haystack or "\u98ce\u9669" in haystack or "\u5371\u9669" in haystack):
            return "safety"
        if ("exit" in haystack or "goal" in haystack) or ("\u51fa\u53e3" in haystack or "\u76ee\u6807" in haystack):
            return "goal"
        if ("dot" in haystack or "pellet" in haystack) or ("\u8c46\u5b50" in haystack or "\u5403\u8c46" in haystack):
            return "dot"
        if ("path" in haystack or "route" in haystack) or ("\u8def\u7ebf" in haystack or "\u8def\u5f84" in haystack):
            return "path"
        if action is not None or any(cue in matched_domain for cue in {"move", "go", "\u8d70", "\u79fb\u52a8"}):
            return "movement"
        return "state"

    @staticmethod
    def _intent_from_frame(
        *,
        topic: str,
        speech_act: str,
        action: Optional[str],
        monster_id: Optional[int],
        negation: bool,
        policy_match: bool,
    ) -> QuestionIntent:
        if policy_match or topic == "policy":
            return QuestionIntent.POLICY_SUMMARY
        if monster_id is not None or topic == "monster":
            return QuestionIntent.MONSTER_INFLUENCE
        if topic == "safety":
            return QuestionIntent.SAFETY_REASON
        if topic == "goal":
            return QuestionIntent.GOAL_REASON
        if topic == "dot":
            return QuestionIntent.DOT_COLLECTION
        if topic == "path":
            return QuestionIntent.PATH_REASON
        if negation and action:
            return QuestionIntent.WHY_NOT_OTHER
        if topic == "movement" or (speech_act == "why" and action):
            return QuestionIntent.WHY_THIS_ACTION
        return QuestionIntent.GENERAL

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
        for pattern in (r"(?:monster|ghost)\s*#?\s*(\d+)", r"怪物\s*#?\s*(\d+)", r"#\s*(\d+)"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def detect_negation(text: str, lang: str) -> bool:
        patterns = _NEGATION_ZH if lang == "zh" else _NEGATION_EN
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def extract_keywords(text: str, lang: str) -> list[str]:
        haystack = text.lower() if lang == "en" else text
        keywords: list[str] = []
        for hints in _RULE_HINTS.values():
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

    @staticmethod
    def _is_clearly_off_topic(text: str, lang: str) -> bool:
        return SemanticMatcher._is_clearly_off_topic(text, lang)

    @staticmethod
    def _looks_like_policy_summary(text: str, lang: str) -> bool:
        haystack = text.lower() if lang == "en" else text
        cues = _POLICY_SUMMARY_CUES_EN if lang == "en" else _POLICY_SUMMARY_CUES_ZH
        return any(cue in haystack for cue in cues)
