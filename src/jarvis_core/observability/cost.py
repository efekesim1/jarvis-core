"""The Cost Engine.

At one agent the API bill is noise. At a hundred agents it is the main
operational risk, so spend is measured per task, per agent, per model and per
day from the first line of code, and a breached budget stops work rather than
quietly continuing.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from ..config import Settings, get_settings
from ..logging import get_logger
from ..models.catalog import ModelCatalog, default_catalog

log = get_logger("cost")

# What a cached prefix read costs relative to a fresh input token.
CACHE_READ_RATE = 0.1


class BudgetScope(StrEnum):
    TASK = "task"
    DAILY = "daily"
    MONTHLY = "monthly"


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend past a budget."""

    def __init__(self, scope: BudgetScope, spent: float, limit: float) -> None:
        super().__init__(
            f"{scope.value} budget exceeded: ${spent:.4f} spent against a ${limit:.2f} limit"
        )
        self.scope = scope
        self.spent = spent
        self.limit = limit


@dataclass(slots=True)
class CostRecord:
    """One priced model call."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent: str | None = None
    task_id: str | None = None
    request_id: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CostEngine:
    """Prices model calls and enforces budgets."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        catalog: ModelCatalog | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or default_catalog()
        self._lock = threading.Lock()
        self._records: list[CostRecord] = []
        self._by_task: dict[str, float] = defaultdict(float)
        self._by_agent: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)
        self._by_day: dict[date, float] = defaultdict(float)
        self._by_request: dict[str, float] = defaultdict(float)

    # -- pricing ----------------------------------------------------------

    def price(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
    ) -> float:
        spec = self.catalog.get(model)
        if spec is None:
            log.warning("no price for model %s; recording it as $0", model)
            return 0.0
        cost = spec.cost_usd(input_tokens, output_tokens)
        # A cached prefix read costs roughly a tenth of a fresh input token.
        cost += cache_read_tokens / 1_000_000 * spec.input_cost_per_mtok * CACHE_READ_RATE
        return cost

    def record(
        self,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent: str | None = None,
        task_id: str | None = None,
        request_id: str | None = None,
        cache_read_tokens: int = 0,
    ) -> CostRecord:
        cost = self.price(model, input_tokens, output_tokens, cache_read_tokens=cache_read_tokens)
        record = CostRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            agent=agent,
            task_id=task_id,
            request_id=request_id,
        )
        with self._lock:
            self._records.append(record)
            self._by_model[model] += cost
            self._by_day[record.at.date()] += cost
            if agent:
                self._by_agent[agent] += cost
            if task_id:
                self._by_task[task_id] += cost
            if request_id:
                self._by_request[request_id] += cost
        return record

    # -- budgets ----------------------------------------------------------

    def spent_today(self) -> float:
        return self._by_day.get(datetime.now(UTC).date(), 0.0)

    def spent_this_month(self) -> float:
        today = datetime.now(UTC).date()
        return sum(
            cost
            for day, cost in self._by_day.items()
            if day.year == today.year and day.month == today.month
        )

    def spent_on_task(self, task_id: str) -> float:
        return self._by_task.get(task_id, 0.0)

    def spent_on_agent(self, agent: str) -> float:
        return self._by_agent.get(agent, 0.0)

    def check_budgets(self, task_id: str | None = None) -> None:
        """Raise :class:`BudgetExceeded` if any budget is already breached."""
        daily = self.spent_today()
        if daily >= self.settings.daily_budget_usd:
            raise BudgetExceeded(BudgetScope.DAILY, daily, self.settings.daily_budget_usd)
        monthly = self.spent_this_month()
        if monthly >= self.settings.monthly_budget_usd:
            raise BudgetExceeded(BudgetScope.MONTHLY, monthly, self.settings.monthly_budget_usd)
        if task_id is not None:
            task_spend = self.spent_on_task(task_id)
            if task_spend >= self.settings.task_budget_usd:
                raise BudgetExceeded(BudgetScope.TASK, task_spend, self.settings.task_budget_usd)

    def budget_pressure(self) -> float:
        """How much of today's budget is gone, from 0.0 to 1.0.

        The router reads this to step down to a cheaper model before the hard
        limit stops work entirely.
        """
        limit = self.settings.daily_budget_usd
        if limit <= 0:
            return 1.0
        return min(1.0, self.spent_today() / limit)

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self._records),
                "total_usd": round(sum(r.cost_usd for r in self._records), 6),
                "today_usd": round(self.spent_today(), 6),
                "month_usd": round(self.spent_this_month(), 6),
                "by_model": {k: round(v, 6) for k, v in sorted(self._by_model.items())},
                "by_agent": {k: round(v, 6) for k, v in sorted(self._by_agent.items())},
                "daily_budget_usd": self.settings.daily_budget_usd,
                "budget_pressure": round(self.budget_pressure(), 4),
            }

    @property
    def records(self) -> list[CostRecord]:
        return list(self._records)
