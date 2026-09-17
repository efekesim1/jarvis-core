"""The model catalog: which models exist, what they cost, what they are good at.

Prices are US dollars per million tokens. Anthropic entries are the verified
first-party API rates. Extra models (OpenAI, Gemini, local) are added through
``config/models.json`` so a new provider never requires a code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..logging import get_logger

log = get_logger("models.catalog")

# The tiers, most capable first. This is the canonical list: the router steps
# down through it under budget pressure, and a spec whose tier is not in it
# matches no tier at all — which for a zero-priced model means it wins every
# fallback, because the fallback is "the cheapest usable model". So a tier that
# is not one of these is corrected rather than carried.
TIERS: tuple[str, ...] = ("frontier", "balanced", "fast")


def normalise_tier(tier: str, *, default: str = "balanced") -> str:
    """Return a tier the router recognises, complaining about one it does not."""
    if tier in TIERS:
        return tier
    log.warning("unknown model tier %r; treating it as %s", tier, default)
    return default


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model and everything the router needs to choose it."""

    id: str
    provider: str
    context_tokens: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    tier: str  # "frontier" | "balanced" | "fast"
    strengths: frozenset[str] = field(default_factory=frozenset)
    supports_server_tools: bool = False
    notes: str = ""

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_cost_per_mtok
            + output_tokens / 1_000_000 * self.output_cost_per_mtok
        )


# Verified Anthropic first-party models and rates.
_BUILTIN: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="claude-opus-5",
        provider="anthropic",
        context_tokens=1_000_000,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=25.0,
        tier="frontier",
        strengths=frozenset({"reasoning", "coding", "planning", "synthesis", "analysis"}),
        supports_server_tools=True,
        notes="Default executive and planning model.",
    ),
    ModelSpec(
        id="claude-sonnet-5",
        provider="anthropic",
        context_tokens=1_000_000,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=10.0,
        tier="balanced",
        strengths=frozenset({"research", "writing", "analysis", "coding"}),
        supports_server_tools=True,
        notes="Workhorse for most agent runs.",
    ),
    ModelSpec(
        id="claude-haiku-4-5",
        provider="anthropic",
        context_tokens=200_000,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=5.0,
        tier="fast",
        strengths=frozenset({"classification", "extraction", "summarization", "routing"}),
        supports_server_tools=False,
        notes="Cheap and fast; use for simple, high-volume steps.",
    ),
    ModelSpec(
        id="claude-fable-5-1",
        provider="anthropic",
        context_tokens=1_000_000,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=50.0,
        tier="frontier",
        strengths=frozenset({"reasoning", "coding", "planning", "long-horizon"}),
        supports_server_tools=True,
        notes="Most capable, most expensive. Reserve for high-value work.",
    ),
)


class ModelCatalog:
    """A lookup of model id to :class:`ModelSpec`."""

    def __init__(self, specs: list[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs if specs is not None else _BUILTIN:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.id] = spec

    def get(self, model_id: str) -> ModelSpec | None:
        return self._specs.get(model_id)

    def require(self, model_id: str) -> ModelSpec:
        spec = self.get(model_id)
        if spec is None:
            known = ", ".join(sorted(self._specs)) or "(none)"
            raise KeyError(f"unknown model {model_id!r}; catalog holds: {known}")
        return spec

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def by_provider(self, provider: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.provider == provider]

    def cheapest(self, candidates: list[ModelSpec] | None = None) -> ModelSpec | None:
        pool = candidates if candidates is not None else self.all()
        if not pool:
            return None
        return min(pool, key=lambda s: s.input_cost_per_mtok + s.output_cost_per_mtok)

    def load_extra(self, path: Path) -> int:
        """Merge additional models from a JSON file. Returns how many loaded."""
        if not path.exists():
            return 0
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read extra model file %s: %s", path, exc)
            return 0
        entries = payload.get("models", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            log.warning("extra model file %s must hold a list of models", path)
            return 0
        count = 0
        for entry in entries:
            try:
                self.register(
                    ModelSpec(
                        id=entry["id"],
                        provider=entry["provider"],
                        context_tokens=int(entry.get("context_tokens", 128_000)),
                        input_cost_per_mtok=float(entry.get("input_cost_per_mtok", 0.0)),
                        output_cost_per_mtok=float(entry.get("output_cost_per_mtok", 0.0)),
                        tier=normalise_tier(entry.get("tier", "balanced")),
                        strengths=frozenset(entry.get("strengths", ())),
                        supports_server_tools=bool(entry.get("supports_server_tools", False)),
                        notes=entry.get("notes", ""),
                    )
                )
                count += 1
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping malformed model entry in %s: %s", path, exc)
        return count


def default_catalog() -> ModelCatalog:
    """The catalog the system uses: built-ins plus anything in config."""
    catalog = ModelCatalog()
    extra = get_settings().extra_models_file
    if extra is not None:
        loaded = catalog.load_extra(Path(extra))
        if loaded:
            log.info("loaded %d extra model(s) from %s", loaded, extra)
    return catalog
