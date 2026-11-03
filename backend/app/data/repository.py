"""In-process dataset repository.

The repository is the only component the analytics layer is allowed to read data
from. It owns:

* lazy, cached loading of the registered CSV datasets;
* safe filter construction from user-supplied analysis filters;
* period and dimension lookups (departments, processes, vendors, months);
* hard row limits, so a pathological query cannot exhaust memory.

It is deliberately not a database. The process holds the CSVs in memory, and
every write path (there is none) is absent by design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.data.loader import (
    DATASET_REGISTRY,
    available_dataset_ids,
    cached_dataset,
    dataset_spec,
    load_manifest,
    manifest_row_count,
    parse_dataset_id,
)
from app.models.enums import DatasetId
from app.schemas.operational import DatasetInfo
from app.utils.errors import DatasetNotFoundError, InsufficientDataError
from app.utils.periods import period_sort_key

logger = get_logger(__name__)

#: Columns a caller may filter on. Anything else is rejected.
FILTERABLE_COLUMNS: dict[DatasetId, frozenset[str]] = {
    DatasetId.OPERATIONS: frozenset(
        {
            "date",
            "period",
            "quarter",
            "department",
            "process",
            "operation_type",
            "status",
            "priority",
            "vendor_id",
        }
    ),
    DatasetId.ORDERS: frozenset(
        {
            "order_date",
            "period",
            "quarter",
            "department",
            "product_category",
            "status",
            "vendor_id",
        }
    ),
    DatasetId.SLA_METRICS: frozenset({"period", "quarter", "department", "process"}),
    DatasetId.VENDORS: frozenset({"vendor_id", "vendor_name", "category", "department"}),
    DatasetId.DEPARTMENTS: frozenset({"department_id", "department_name", "operational_area"}),
}

#: Human readable aliases so a plan can request "department=DEP-102" by name.
COLUMN_ALIASES: dict[str, str] = {
    "department_id": "department",
    "department_name": "department",
    "order_date": "date",
}


@dataclass(frozen=True, slots=True)
class FilterSpec:
    """One validated equality/inclusion filter."""

    column: str
    values: tuple[str, ...]
    include_nulls: bool = False


class DatasetRepository:
    """Read access to the loaded synthetic datasets."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._data_dir = self._settings.data_dir
        self._manifest = load_manifest(self._data_dir)
        self._loaded: dict[DatasetId, pd.DataFrame] = {}

    # -- catalogue -----------------------------------------------------
    @property
    def data_dir(self) -> object:
        return self._data_dir

    @property
    def max_rows(self) -> int:
        return self._settings.max_analysis_rows

    def available_datasets(self) -> list[DatasetId]:
        return available_dataset_ids(self._data_dir)

    def dataset_info(self, dataset_id: DatasetId) -> DatasetInfo:
        spec = dataset_spec(dataset_id)
        frame = self.frame(dataset_id)
        return DatasetInfo(
            dataset_id=dataset_id,
            file_name=spec.file_name,
            description=spec.description,
            grain=spec.grain,
            primary_key=spec.primary_key,
            time_column=spec.time_column,
            dimensions=spec.dimensions,
            measures=spec.measures,
            row_count=len(frame),
            column_count=len(frame.columns),
            loaded=True,
        )

    def catalogue(self) -> list[DatasetInfo]:
        infos: list[DatasetInfo] = []
        for dataset_id in self.available_datasets():
            try:
                infos.append(self.dataset_info(dataset_id))
            except (DatasetNotFoundError, InsufficientDataError):
                logger.warning("dataset_info_failed", extra={"dataset_id": dataset_id.value})
        return infos

    # -- frame access --------------------------------------------------
    def frame(self, dataset_id: DatasetId | str) -> pd.DataFrame:
        """Return the cached, read-only DataFrame for a dataset."""
        resolved = parse_dataset_id(
            dataset_id.value if hasattr(dataset_id, "value") else str(dataset_id)
        )
        if resolved in self._loaded:
            return self._loaded[resolved]
        try:
            frame = cached_dataset(resolved, str(self._data_dir))
        except DatasetNotFoundError:
            raise
        self._loaded[resolved] = frame
        return frame

    def working_copy(self, dataset_id: DatasetId | str) -> pd.DataFrame:
        """A writable copy, safe to filter and mutate."""
        return self.frame(dataset_id).copy()

    def require(self, dataset_ids: Sequence[DatasetId]) -> dict[DatasetId, pd.DataFrame]:
        """Load several datasets at once, raising if any is missing."""
        return {dataset_id: self.frame(dataset_id) for dataset_id in dataset_ids}

    def row_count(self, dataset_id: DatasetId) -> int:
        return len(self.frame(dataset_id))

    def apply_filters(
        self,
        frame: pd.DataFrame,
        dataset_id: DatasetId,
        filters: Sequence[FilterSpec],
    ) -> pd.DataFrame:
        """Apply validated filters and enforce the configured row limit."""
        if not filters:
            return frame
        allowed = FILTERABLE_COLUMNS[dataset_id]
        result = frame
        for spec in filters:
            column = COLUMN_ALIASES.get(spec.column, spec.column)
            if column not in allowed:
                raise DatasetNotFoundError(
                    f"column '{spec.column}' cannot be filtered on dataset "
                    f"'{dataset_id.value}'; allowed columns: {', '.join(sorted(allowed))}"
                )
            if column not in result.columns:
                continue
            mask = result[column].astype("string").isin(list(spec.values))
            if spec.include_nulls:
                mask = mask | result[column].isna()
            result = result.loc[mask]

        if len(result) > self.max_rows:
            raise InsufficientDataError(
                f"filter selected {len(result)} rows which exceeds the configured limit "
                f"of {self.max_rows}; narrow the request"
            )
        return result

    # -- reference lookups ---------------------------------------------
    @cached_property
    def departments(self) -> pd.DataFrame:
        return self.frame(DatasetId.DEPARTMENTS)

    @cached_property
    def vendors(self) -> pd.DataFrame:
        return self.frame(DatasetId.VENDORS)

    @cached_property
    def available_periods(self) -> list[str]:
        frame = self.frame(DatasetId.OPERATIONS)
        months = set(frame["period"].dropna().astype(str))
        order_months = set(self.frame(DatasetId.ORDERS)["period"].dropna().astype(str))
        months |= order_months
        return sorted(months, key=period_sort_key)

    @cached_property
    def available_period_set(self) -> frozenset[str]:
        return frozenset(self.available_periods)

    @cached_property
    def available_quarters(self) -> list[str]:
        frame = self.frame(DatasetId.OPERATIONS)
        quarters = set(frame["quarter"].dropna().astype(str))
        quarters |= set(self.frame(DatasetId.ORDERS)["quarter"].dropna().astype(str))
        return sorted(quarters)

    @cached_property
    def department_ids(self) -> list[str]:
        frame = self.frame(DatasetId.OPERATIONS)
        return sorted(str(value) for value in frame["department"].dropna().unique())

    @cached_property
    def department_names(self) -> dict[str, str]:
        frame = self.departments
        return {
            str(row["department_id"]): str(row["department_name"]) for _, row in frame.iterrows()
        }

    @cached_property
    def process_names(self) -> list[str]:
        frame = self.frame(DatasetId.OPERATIONS)
        return sorted(str(value) for value in frame["process"].dropna().unique())

    @cached_property
    def vendor_ids(self) -> list[str]:
        frame = self.frame(DatasetId.OPERATIONS)
        values = set(frame["vendor_id"].dropna().astype(str))
        values |= set(self.frame(DatasetId.ORDERS)["vendor_id"].dropna().astype(str))
        return sorted(value for value in values if value)

    @cached_property
    def product_categories(self) -> list[str]:
        frame = self.frame(DatasetId.ORDERS)
        return sorted(str(value) for value in frame["product_category"].dropna().unique())

    @cached_property
    def date_range(self) -> tuple[str, str]:
        frame = self.frame(DatasetId.OPERATIONS)
        dates = frame["date"].dropna()
        if dates.empty:
            return ("", "")
        return (dates.min().date().isoformat(), dates.max().date().isoformat())

    @cached_property
    def manifest_row_counts(self) -> Mapping[str, int]:
        if self._manifest is None:
            return {}
        counts: dict[str, int] = {}
        for _, row in self._manifest.iterrows():
            try:
                counts[str(row["dataset_id"])] = int(str(row["row_count"]))
            except (KeyError, TypeError, ValueError):  # pragma: no cover
                continue
        return counts

    def declared_row_count(self, dataset_id: DatasetId) -> int | None:
        return manifest_row_count(self._data_dir, dataset_id)

    # -- convenience views ---------------------------------------------
    def operations(self, filters: Sequence[FilterSpec] = ()) -> pd.DataFrame:
        return self.apply_filters(self.frame(DatasetId.OPERATIONS), DatasetId.OPERATIONS, filters)

    def orders(self, filters: Sequence[FilterSpec] = ()) -> pd.DataFrame:
        return self.apply_filters(self.frame(DatasetId.ORDERS), DatasetId.ORDERS, filters)

    def sla_metrics(self, filters: Sequence[FilterSpec] = ()) -> pd.DataFrame:
        return self.apply_filters(self.frame(DatasetId.SLA_METRICS), DatasetId.SLA_METRICS, filters)

    def dataset_columns(self, dataset_id: DatasetId) -> tuple[str, ...]:
        return tuple(DATASET_REGISTRY[dataset_id].dtypes.keys())

    def reset_caches(self) -> None:
        """Drop derived caches (used by tests that swap datasets)."""
        self._loaded.clear()
        for attribute in (
            "departments",
            "vendors",
            "available_periods",
            "available_period_set",
            "available_quarters",
            "department_ids",
            "department_names",
            "process_names",
            "vendor_ids",
            "product_categories",
            "date_range",
            "manifest_row_counts",
        ):
            self.__dict__.pop(attribute, None)


_repository: DatasetRepository | None = None


def get_repository() -> DatasetRepository:
    """Process-wide repository singleton."""
    global _repository
    if _repository is None:
        _repository = DatasetRepository()
        logger.info(
            "repository_ready",
            extra={
                "datasets": [item.value for item in _repository.available_datasets()],
                "periods": len(_repository.available_periods),
            },
        )
    return _repository


def reset_repository() -> None:
    """Drop the singleton (used by tests)."""
    global _repository
    if _repository is not None:
        _repository.reset_caches()
    _repository = None
