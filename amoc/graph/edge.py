from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _EMB_MODEL: Optional[SentenceTransformer] = None
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    np = None
    _EMB_MODEL = None

if TYPE_CHECKING:
    from amoc.graph.node import Node


def _maybe_embed(text: str) -> Optional["np.ndarray"]:
    global _EMB_MODEL
    if SentenceTransformer is None or np is None:
        return None
    if _EMB_MODEL is None:
        try:
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB_MODEL = None
            return None
    try:
        vecs = _EMB_MODEL.encode([text], normalize_embeddings=True)
        return vecs[0]
    except Exception:
        return None


class Edge:
    def __init__(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        forget_score: int,
        active: bool = True,
        created_at_sentence: Optional[int] = None,
    ) -> None:
        self.source_node: Node = source_node
        self.dest_node: Node = dest_node
        self.active: bool = active
        self.label: str = label
        self.forget_score: int = forget_score
        self.similarity_threshold = 0.8
        self.embedding: Optional["np.ndarray"] = _maybe_embed(label)
        self.created_at_sentence: Optional[int] = created_at_sentence

    def fade_away(self) -> None:
        self.forget_score -= 1
        if self.forget_score <= 0:
            self.active = False

    def is_similar(self, other_edge: "Edge") -> bool:
        a = (self.label or "").strip().lower()
        b = (other_edge.label or "").strip().lower()
        if a == b:
            return True

        if self.embedding is not None and other_edge.embedding is not None:
            cos = float(np.dot(self.embedding, other_edge.embedding))
            if cos >= self.similarity_threshold:
                return True

        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold

    def __eq__(self, other: "Edge") -> bool:
        return (
            self.source_node == other.source_node
            and self.dest_node == other.dest_node
            and self.label == other.label
        )

    def __hash__(self) -> int:
        return hash((self.source_node, self.dest_node, self.label))

    def __str__(self) -> str:
        return f"{self.source_node.get_text_representer()} --{self.label} ({'active' if self.active else 'inactive'})--> {self.dest_node.get_text_representer()} ({self.forget_score})"

    def __repr__(self) -> str:
        return self.__str__()
