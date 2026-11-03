"""Pydantic schemas for raw operational records and dataset metadata.

``OperationalRecord`` is the typed representation of a row in
``data/operations.csv``. Loading a CSV into these models is what enforces the
data contract: a malformed CSV fails loudly at the boundary instead of
propagating NaN into the analytics layer.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import DatasetId


class OperationalStatus(str):
    """Allowed values of ``operations.status``."""


class OperationalRecord(BaseModel):
    """One operational record: a department/process on a single day."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=64)
    date: dt.date
    period: str = Field(pattern=r"^\d{4}-\d{2}$")
    quarter: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    department: str = Field(min_length=1, max_length=32)
    process: str = Field(min_length=1, max_length=96)
    operation_type: str = Field(min_length=1, max_length=64)
    status: str = Field(pattern=r"^(completed|at_risk|delayed)$")
    priority: str = Field(pattern=r"^(standard|expedited|critical)$")
    completed_operations: int = Field(ge=0)
    delayed_operations: int = Field(ge=0)
    processing_time_hours: float = Field(ge=0.0, le=1_000.0)
    queue_time_hours: float = Field(ge=0.0, le=1_000.0)
    total_cycle_time_hours: float = Field(ge=0.0, le=1_000.0)
    employee_count: int = Field(ge=0, le=100_000)
    vendor_id: str | None = Field(default=None, max_length=32)
    backlog_open_items: float = Field(ge=0.0, le=10_000_000.0)

    @field_validator("vendor_id", mode="before")
    @classmethod
    def _blank_vendor_is_null(cls, value: object) -> object:
        """An empty CSV cell means 'internal', not a vendor named ''."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("delayed_operations")
    @classmethod
    def _delayed_within_completed(cls, value: int, info: object) -> int:
        completed = getattr(info, "data", {}).get("completed_operations")
        if isinstance(completed, int) and value > completed:
            raise ValueError("delayed_operations cannot exceed completed_operations")
        return value

    @property
    def delay_rate(self) -> float:
        """Share of completed operations that were late. 0.0 when no volume."""
        if self.completed_operations <= 0:
            return 0.0
        return self.delayed_operations / self.completed_operations * 100.0

    @property
    def queue_contribution(self) -> float:
        """Share of total cycle time spent waiting. 0.0 when cycle time is 0."""
        if self.total_cycle_time_hours <= 0:
            return 0.0
        return self.queue_time_hours / self.total_cycle_time_hours * 100.0


class OrderRecord(BaseModel):
    """One order from ``data/orders.csv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(min_length=1, max_length=64)
    order_date: dt.date
    period: str = Field(pattern=r"^\d{4}-\d{2}$")
    quarter: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    department: str = Field(min_length=1, max_length=32)
    product_category: str = Field(min_length=1, max_length=64)
    order_value: float = Field(ge=0.0, le=1e9)
    promised_time_hours: float = Field(ge=0.0, le=10_000.0)
    actual_time_hours: float = Field(ge=0.0, le=10_000.0)
    fulfillment_time_hours: float = Field(ge=0.0, le=10_000.0)
    transit_time_hours: float = Field(ge=0.0, le=10_000.0)
    status: str = Field(pattern=r"^(delivered|delivered_late|returned|cancelled)$")
    vendor_id: str | None = Field(default=None, max_length=32)
    delay_hours: float = Field(ge=0.0, le=10_000.0)

    @field_validator("vendor_id", mode="before")
    @classmethod
    def _blank_vendor_is_null(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_late(self) -> bool:
        return self.status == "delivered_late"


class SlaMetricRecord(BaseModel):
    """One monthly SLA rollup row from ``data/sla_metrics.csv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    period: str = Field(pattern=r"^\d{4}-\d{2}$")
    quarter: str = Field(pattern=r"^\d{4}-Q[1-4]$")
    department: str = Field(min_length=1, max_length=32)
    process: str = Field(min_length=1, max_length=96)
    total_cases: int = Field(ge=0)
    sla_target_hours: float = Field(gt=0.0, le=10_000.0)
    average_cycle_time_hours: float = Field(ge=0.0, le=10_000.0)
    average_queue_time_hours: float = Field(ge=0.0, le=10_000.0)
    sla_met_cases: int = Field(ge=0)
    sla_breached_cases: int = Field(ge=0)
    sla_compliance_percentage: float = Field(ge=0.0, le=100.0)
    backlog_open_items: float = Field(ge=0.0)

    @field_validator("sla_breached_cases")
    @classmethod
    def _cases_balance(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        total = data.get("total_cases")
        met = data.get("sla_met_cases")
        if isinstance(total, int) and isinstance(met, int) and value != total - met:
            raise ValueError("sla_breached_cases must equal total_cases - sla_met_cases")
        return value


class VendorRecord(BaseModel):
    """One vendor profile from ``data/vendors.csv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vendor_id: str = Field(min_length=1, max_length=32)
    vendor_name: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    department: str = Field(min_length=1, max_length=32)
    monthly_volume: float = Field(ge=0.0)
    average_processing_time: float = Field(ge=0.0, le=10_000.0)
    delay_rate: float = Field(ge=0.0, le=100.0)
    sla_compliance: float = Field(ge=0.0, le=100.0)


class DepartmentRecord(BaseModel):
    """One department from ``data/departments.csv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    department_id: str = Field(min_length=1, max_length=32)
    department_name: str = Field(min_length=1, max_length=128)
    operational_area: str = Field(min_length=1, max_length=96)
    headcount: int = Field(ge=0, le=100_000)


class DatasetInfo(BaseModel):
    """Metadata describing a registered dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: DatasetId
    file_name: str
    description: str
    grain: str
    primary_key: str
    time_column: str | None = None
    dimensions: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    loaded: bool = False


class DatasetSummary(BaseModel):
    """Aggregated profile of a loaded dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: DatasetId
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[str, ...]
    row_count_matches_manifest: bool
    period_start: str | None = None
    period_end: str | None = None
    distinct_departments: int = Field(ge=0)
    distinct_processes: int = Field(ge=0)
    distinct_vendors: int = Field(ge=0)
    numeric_measures: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    completeness: dict[str, float] = Field(default_factory=dict)
