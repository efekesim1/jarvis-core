"""Process-wide configuration.

Every secret is read from the environment (or a local ``.env`` file that is
never committed). Nothing in this repository may contain an API key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    """Runtime settings, loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- environment -----------------------------------------------------
    environment: str = Field(
        default="sandbox",
        description="One of: sandbox, staging, production. Sandbox never receives production secrets.",
    )
    log_level: str = "INFO"
    data_dir: Path = Field(default=PROJECT_ROOT / "var")

    # --- model providers -------------------------------------------------
    # Keys are read from the provider-native variables so the official SDKs
    # keep working unchanged. They are never written to disk or logs.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")

    default_model: str = "claude-opus-5"
    fast_model: str = "claude-haiku-4-5"
    balanced_model: str = "claude-sonnet-5"
    frontier_model: str = "claude-opus-5"

    # --- storage ---------------------------------------------------------
    # Where durable state lives. Unset means everything stays in memory, which
    # is what the tests and a quick CLI run want. Set it and tasks, approvals,
    # memory, audit and cost survive a restart and can be shared between two
    # processes on this machine.
    database_path: Path | None = None

    # The directory file tools may read. Everything outside it is refused,
    # symlinks included. Unset means no file tools are registered at all.
    workspace_dir: Path | None = Field(default=PROJECT_ROOT)

    # --- a second vendor -------------------------------------------------
    #
    # NVIDIA's NIM endpoint speaks the Chat Completions protocol, so it reuses
    # the OpenAI provider with a different base URL. A second vendor is the one
    # thing that makes routing *quality* measurable: with one, every model in
    # the catalogue succeeds and fails together and there is nothing to compare.
    #
    # The model must be named explicitly, like OPENAI_MODEL, because a
    # catalogue of hundreds changes faster than this code should.
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_model: str | None = Field(default=None, alias="NVIDIA_MODEL")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # xAI and Gemini, both reached over Chat Completions for the same reason:
    # one request format, one usage shape, one set of error strings for
    # `classify_fault` to know. Gemini has a native API this deliberately does
    # not use — see `GeminiProvider` for what that costs.
    #
    # Each needs its model named explicitly, like the others, because a
    # catalogue changes faster than this code should.
    xai_api_key: str | None = Field(default=None, alias="XAI_API_KEY")
    xai_model: str | None = Field(default=None, alias="XAI_MODEL")
    xai_base_url: str = "https://api.x.ai/v1"

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str | None = Field(default=None, alias="GEMINI_MODEL")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # --- the free tier: whatever is running on this machine ---------------
    #
    # Ollama needs no key, costs no dollars and works with the network down,
    # which makes it the one provider that is either there or not there rather
    # than authorised or unauthorised. It is reached over its native API with
    # the standard library, deliberately: the whole point of the local tier is
    # that it works with nothing installed, and requiring an SDK to talk to
    # localhost would put the cheapest path behind the same extras as the
    # expensive ones.
    #
    # `ollama_model` is optional, unlike every key-bearing provider above,
    # because the daemon can be *asked* what it has. Naming one pins the
    # routing; leaving it empty lets every pulled model become a usable $0
    # entry in the catalogue.
    ollama_enabled: bool = Field(default=True, alias="OLLAMA_ENABLED")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str | None = Field(default=None, alias="OLLAMA_MODEL")
    # Local models enter the `fast` tier, which is what SIMPLE work asks for
    # and what every other tier steps down to under budget pressure. A 7B model
    # is not a frontier model and claiming otherwise in the catalogue would
    # route planning to it. Override only if you know your local model is
    # stronger than that.
    ollama_tier: str = Field(default="fast", alias="OLLAMA_TIER")
    # How long a reachability answer is trusted. The probe is one HTTP call to
    # localhost, but `available()` is consulted on every routing decision, so
    # it is cached rather than repeated.
    ollama_probe_ttl_s: float = 30.0

    # --- registries ------------------------------------------------------
    agents_dir: Path = Field(default=PROJECT_ROOT / "agents")
    skills_dir: Path = Field(default=PROJECT_ROOT / "skills")
    extra_models_file: Path | None = Field(default=PROJECT_ROOT / "config" / "models.json")

    # --- budgets (USD) ---------------------------------------------------
    daily_budget_usd: float = 10.0
    monthly_budget_usd: float = 200.0
    task_budget_usd: float = 2.0
    # --- the real-run governor -------------------------------------------
    #
    # Four independent ceilings on what one real-model sweep may do, because
    # one of them is always the wrong one to have chosen. Money is the obvious
    # limit and the slowest to notice; a runaway retry loop hits the call count
    # first, a single agent that will not stop generating hits the token count
    # first, and a hung provider hits the clock first. Whichever binds first
    # stops the sweep.
    #
    # Deliberately smaller than a day's budget: `jarvis verify --real` reads
    # like a status check and is in fact twenty-seven paid model calls, so the
    # defaults are numbers nobody would mind having spent by accident.
    real_validation_budget_usd: float = 1.0
    real_run_max_calls: int = 60
    real_run_max_tokens: int = 400_000
    real_run_timeout_seconds: float = 1800.0

    # --- execution limits ------------------------------------------------
    max_parallel_agents: int = 6
    default_agent_timeout_s: float = 180.0
    max_retries_per_task: int = 2
    retry_budget_per_request: int = 6
    # Hard ceilings on how far one request may expand. Without these an
    # autonomous run has no upper bound on what it can spend.
    max_agent_hops: int = 12
    max_plan_waves: int = 6

    # --- context budget --------------------------------------------------
    # An agent reads only what its own step depends on, trimmed to these sizes.
    # Passing a whole conversation to every agent is how a budget disappears.
    max_finding_chars: int = 1500
    max_findings: int = 6
    max_context_chars: int = 12000

    # --- caching ---------------------------------------------------------
    # Identical work inside the window is answered from the cache instead of
    # being paid for twice.
    cache_enabled: bool = True
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 512

    # --- approval --------------------------------------------------------
    require_approval_above: str = Field(
        default="medium",
        description="Risk level at and above which a human must approve: low, medium, high.",
    )

    @field_validator("database_path", "workspace_dir", "extra_models_file", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat an empty value as absent.

        ``JARVIS_DATABASE_PATH=`` in a .env should mean "no database", not a path
        whose name is the empty string.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def provider_key(self, provider: str) -> str | None:
        return {"anthropic": self.anthropic_api_key, "openai": self.openai_api_key}.get(provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear the settings cache. Used by tests after patching the environment."""
    get_settings.cache_clear()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
