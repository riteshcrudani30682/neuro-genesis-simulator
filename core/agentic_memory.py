"""Persistent, bounded and scored memory for Agentic Core V1.

The store keeps compact JSON-safe snapshots rather than arbitrary Python objects.
Retrieval combines context overlap, outcome quality and recency so specialists see
only a few relevant past experiences instead of an ever-growing transcript.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping

from core.agentic import Experience, Observation


def _safe(value: Any, *, max_text: int = 500) -> Any:
    """Convert common values to bounded JSON-safe representations."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, str):
            return value[:max_text]
        return value
    if isinstance(value, Mapping):
        return {str(k)[:100]: _safe(v, max_text=max_text) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_safe(v, max_text=max_text) for v in value[:64]]
    if is_dataclass(value):
        return _safe(asdict(value), max_text=max_text)
    if hasattr(value, "name"):
        return str(value.name)[:max_text]
    return repr(value)[:max_text]


def _tokens(value: Any) -> set[str]:
    text = json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return {part for part in cleaned.split() if len(part) >= 2}


@dataclass(frozen=True)
class MemoryHit:
    score: float
    action: Any
    selected_agent: str | None
    decision_confidence: float
    verification_ok: bool
    verification_score: float
    rationale: str
    context: Mapping[str, Any]

    def to_context(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "action": self.action,
            "selected_agent": self.selected_agent,
            "decision_confidence": self.decision_confidence,
            "verification_ok": self.verification_ok,
            "verification_score": self.verification_score,
            "rationale": self.rationale,
            "context": dict(self.context),
        }


class PersistentExperienceStore:
    """Bounded disk-backed memory with deterministic relevance scoring."""

    SCHEMA = 1

    def __init__(self, path: str | Path, *, capacity: int = 1000, top_k: int = 5) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 100_000:
            raise ValueError("capacity must be 1..100000")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be 1..20")
        self.path = Path(path)
        self.capacity = capacity
        self.top_k = top_k
        self._items: List[Dict[str, Any]] = []
        if self.path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self._items)

    def _snapshot(self, experience: Experience) -> Dict[str, Any]:
        return {
            "state": _safe(experience.observation.state),
            "context": _safe(experience.observation.context),
            "action": _safe(experience.decision.action),
            "selected_agent": experience.decision.selected_agent,
            "decision_confidence": float(experience.decision.confidence),
            "rationale": experience.decision.rationale[:500],
            "verification_ok": bool(experience.verification.ok),
            "verification_score": float(experience.verification.score),
            "verification_details": experience.verification.details[:500],
        }

    def remember(self, experience: Experience) -> None:
        score = float(experience.verification.score)
        if not math.isfinite(score):
            raise ValueError("verification score must be finite")
        self._items.append(self._snapshot(experience))
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]
        self.save()

    def retrieve(self, observation: Observation, *, top_k: int | None = None) -> List[MemoryHit]:
        limit = self.top_k if top_k is None else top_k
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("top_k must be 1..20")
        query_tokens = _tokens({"state": observation.state, "context": observation.context})
        hits: List[MemoryHit] = []
        total = max(1, len(self._items))
        for index, item in enumerate(self._items):
            memory_tokens = _tokens({"state": item["state"], "context": item["context"]})
            union = query_tokens | memory_tokens
            overlap = (len(query_tokens & memory_tokens) / len(union)) if union else 0.0
            quality = max(0.0, min(1.0, float(item["verification_score"])))
            if not item["verification_ok"]:
                quality *= 0.35
            recency = (index + 1) / total
            score = 0.65 * overlap + 0.25 * quality + 0.10 * recency
            hits.append(
                MemoryHit(
                    score=score,
                    action=item["action"],
                    selected_agent=item.get("selected_agent"),
                    decision_confidence=float(item["decision_confidence"]),
                    verification_ok=bool(item["verification_ok"]),
                    verification_score=float(item["verification_score"]),
                    rationale=str(item.get("rationale", "")),
                    context=item.get("context", {}),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def context(self, observation: Observation) -> List[Dict[str, Any]]:
        return [hit.to_context() for hit in self.retrieve(observation)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA,
            "capacity": self.capacity,
            "top_k": self.top_k,
            "items": self._items,
        }
        temp = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temp = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            temp.replace(self.path)
        finally:
            if temp and temp.exists():
                temp.unlink()

    def _load(self) -> None:
        if self.path.stat().st_size > 32_000_000:
            raise ValueError("agentic memory file is too large")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != self.SCHEMA:
            raise ValueError("agentic memory schema mismatch")
        items = data.get("items")
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError("invalid agentic memory items")
        self._items = items
