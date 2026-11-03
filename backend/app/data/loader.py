"""Dataset registry and safe CSV loading.

Security properties of this module:

* Dataset identifiers are resolved through a fixed registry. A caller can never
  supply a filesystem path, glob, or traversal sequence.
* The resolved path is asserted to be inside the configured data directory, so a
  registry bug cannot escape the sandbox.
* CSVs are read with an explicit dtype map so that ids such as ``DEP-101`` and
  periods such as ``2025-01`` are never coerced to numbers.
* Values are coerced to numeric with a NaN policy; NaN never escapes this module.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.config import DEFAULT_DATA_DIR, Settings
from app.core.logging import get_logger
from app.models.enums import DatasetId
from app.utils.errors import ConfigurationError, DatasetLoadError, DatasetNotFoundError

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static description of a registered CSV dataset."""

    dataset_id: DatasetId
    file_name: str
    record_model: str
    description: str
    grain: str
    primary_key: str
    time_column: str | None
    dimensions: tuple[str, ...]
    measures: tuple[str, ...]
    #: column -> pandas dtype. Absent columns are tolerated but not coerced.
    dtypes: dict[str, str]


_OPERATIONS_COLUMNS = (
    "operation_id",
    "date",
    "period",
    "quarter",
    "department",
    "process",
    "operation_type",
    "status",
    "priority",
    "completed_operations",
    "delayed_operations",
    "processing_time_hours",
    "queue_time_hours",
    "total_cycle_time_hours",
    "employee_count",
    "vendor_id",
    "backlog_open_items",
)
_ORDER_COLUMNS = (
    "order_id",
    "order_date",
    "period",
    "quarter",
    "department",
    "product_category",
    "order_value",
    "promised_time_hours",
    "actual_time_hours",
    "fulfillment_time_hours",
    "transit_time_hours",
    "status",
    "vendor_id",
    "delay_hours",
)
_SLA_COLUMNS = (
    "period",
    "quarter",
    "department",
    "process",
    "total_cases",
    "sla_target_hours",
    "average_cycle_time_hours",
    "average_queue_time_hours",
    "sla_met_cases",
    "sla_breached_cases",
    "sla_compliance_percentage",
    "backlog_open_items",
)
_VENDOR_COLUMNS = (
    "vendor_id",
    "vendor_name",
    "category",
    "department",
    "monthly_volume",
    "average_processing_time",
    "delay_rate",
    "sla_compliance",
)
_DEPARTMENT_COLUMNS = ("department_id", "department_name", "operational_area", "headcount")


DATASET_REGISTRY: dict[DatasetId, DatasetSpec] = {
    DatasetId.OPERATIONS: DatasetSpec(
        dataset_id=DatasetId.OPERATIONS,
        file_name="operations.csv",
        record_model="OperationalRecord",
        description=(
            "Daily operational throughput, cycle time, queue and backlog by department and process."
        ),
        grain="one row per department-process-day",
        primary_key="operation_id",
        time_column="date",
        dimensions=(
            "department",
            "process",
            "operation_type",
            "status",
            "priority",
            "vendor_id",
            "period",
            "quarter",
        ),
        measures=(
            "completed_operations",
            "delayed_operations",
            "processing_time_hours",
            "queue_time_hours",
            "total_cycle_time_hours",
            "employee_count",
            "backlog_open_items",
        ),
        dtypes={
            "operation_id": "string",
            "date": "string",
            "period": "string",
            "quarter": "string",
            "department": "string",
            "process": "string",
            "operation_type": "string",
            "status": "string",
            "priority": "string",
            "completed_operations": "Int64",
            "delayed_operations": "Int64",
            "processing_time_hours": "float64",
            "queue_time_hours": "float64",
            "total_cycle_time_hours": "float64",
            "employee_count": "Int64",
            "vendor_id": "string",
            "backlog_open_items": "float64",
        },
    ),
    DatasetId.ORDERS: DatasetSpec(
        dataset_id=DatasetId.ORDERS,
        file_name="orders.csv",
        record_model="OrderRecord",
        description=(
            "Order-level fulfilment detail for fulfilment-time, promise and "
            "vendor transit analysis."
        ),
        grain="one row per order",
        primary_key="order_id",
        time_column="order_date",
        dimensions=("department", "product_category", "status", "vendor_id", "period", "quarter"),
        measures=(
            "order_value",
            "promised_time_hours",
            "actual_time_hours",
            "fulfillment_time_hours",
            "delay_hours",
        ),
        dtypes={
            "order_id": "string",
            "order_date": "string",
            "period": "string",
            "quarter": "string",
            "department": "string",
            "product_category": "string",
            "order_value": "float64",
            "promised_time_hours": "float64",
            "actual_time_hours": "float64",
            "fulfillment_time_hours": "float64",
            "transit_time_hours": "float64",
            "status": "string",
            "vendor_id": "string",
            "delay_hours": "float64",
        },
    ),
    DatasetId.SLA_METRICS: DatasetSpec(
        dataset_id=DatasetId.SLA_METRICS,
        file_name="sla_metrics.csv",
        record_model="SlaMetricRecord",
        description="Monthly SLA rollup aggregated from operations.csv.",
        grain="one row per department-process-month",
        primary_key="department|process|period",
        time_column="period",
        dimensions=("department", "process", "period", "quarter"),
        measures=(
            "total_cases",
            "sla_target_hours",
            "average_cycle_time_hours",
            "average_queue_time_hours",
            "sla_met_cases",
            "sla_breached_cases",
            "sla_compliance_percentage",
            "backlog_open_items",
        ),
        dtypes={
            "period": "string",
            "quarter": "string",
            "department": "string",
            "process": "string",
            "total_cases": "Int64",
            "sla_target_hours": "float64",
            "average_cycle_time_hours": "float64",
            "average_queue_time_hours": "float64",
            "sla_met_cases": "Int64",
            "sla_breached_cases": "Int64",
            "sla_compliance_percentage": "float64",
            "backlog_open_items": "float64",
        },
    ),
    DatasetId.VENDORS: DatasetSpec(
        dataset_id=DatasetId.VENDORS,
        file_name="vendors.csv",
        record_model="VendorRecord",
        description="Vendor reference profile aggregated from vendor-linked operational records.",
        grain="one row per vendor",
        primary_key="vendor_id",
        time_column=None,
        dimensions=("vendor_id", "vendor_name", "category", "department"),
        measures=("monthly_volume", "average_processing_time", "delay_rate", "sla_compliance"),
        dtypes={
            "vendor_id": "string",
            "vendor_name": "string",
            "category": "string",
            "department": "string",
            "monthly_volume": "float64",
            "average_processing_time": "float64",
            "delay_rate": "float64",
            "sla_compliance": "float64",
        },
    ),
    DatasetId.DEPARTMENTS: DatasetSpec(
        dataset_id=DatasetId.DEPARTMENTS,
        file_name="departments.csv",
        record_model="DepartmentRecord",
        description="Department reference data and allocated headcount.",
        grain="one row per department",
        primary_key="department_id",
        time_column=None,
        dimensions=("department_id", "department_name", "operational_area"),
        measures=("headcount",),
        dtypes={
            "department_id": "string",
            "department_name": "string",
            "operational_area": "string",
            "headcount": "Int64",
        },
    ),
}

REQUIRED_COLUMNS: dict[DatasetId, tuple[str, ...]] = {
    DatasetId.OPERATIONS: _OPERATIONS_COLUMNS,
    DatasetId.ORDERS: _ORDER_COLUMNS,
    DatasetId.SLA_METRICS: _SLA_COLUMNS,
    DatasetId.VENDORS: _VENDOR_COLUMNS,
    DatasetId.DEPARTMENTS: _DEPARTMENT_COLUMNS,
}


def parse_dataset_id(value: str) -> DatasetId:
    """Resolve a user-supplied dataset identifier against the registry."""
    try:
        return DatasetId(value.strip().lower())
    except (ValueError, AttributeError) as exc:
        allowed = ", ".join(sorted(item.value for item in DatasetId))
        raise DatasetNotFoundError(
            f"unknown dataset '{value}'; available datasets: {allowed}"
        ) from exc


def resolve_dataset_path(dataset_id: DatasetId, data_dir: Path | None = None) -> Path:
    """Return the absolute CSV path for a registered dataset.

    The path is built from the registry and the configured data directory only.
    There is no code path that accepts a caller-supplied path.
    """
    spec = DATASET_REGISTRY[dataset_id]
    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    try:
        base_resolved = base.resolve()
    except OSError as exc:  # pragma: no cover - unusual filesystem failure
        raise ConfigurationError(f"data directory '{base}' could not be resolved") from exc

    candidate = (base_resolved / spec.file_name).resolve()
    if base_resolved not in candidate.parents:
        raise ConfigurationError(
            f"dataset path for '{dataset_id}' resolves outside the configured data directory"
        )
    if not candidate.is_file():
        raise DatasetNotFoundError(
            f"dataset file '{spec.file_name}' was not found in '{base_resolved}'"
        )
    return candidate


def load_dataset(dataset_id: DatasetId, data_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate one dataset into a DataFrame."""
    spec = DATASET_REGISTRY[dataset_id]
    path = resolve_dataset_path(dataset_id, data_dir)
    try:
        frame = pd.read_csv(path, dtype=spec.dtypes, keep_default_na=True)
    except FileNotFoundError as exc:  # pragma: no cover - guarded above
        raise DatasetNotFoundError(f"dataset file '{spec.file_name}' was not found") from exc
    except Exception as exc:
        raise DatasetLoadError(f"dataset '{dataset_id}' could not be read: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS[dataset_id] if c not in frame.columns]
    if missing:
        raise DatasetLoadError(
            f"dataset '{dataset_id}' is missing required columns: {', '.join(missing)}"
        )

    frame = _normalise(frame, dataset_id)
    logger.info(
        "dataset_loaded",
        extra={"dataset_id": dataset_id.value, "rows": len(frame), "file_name": spec.file_name},
    )
    return frame


def _normalise(frame: pd.DataFrame, dataset_id: DatasetId) -> pd.DataFrame:
    """Normalise date columns and enforce numeric sanity."""
    spec = DATASET_REGISTRY[dataset_id]
    normalised = frame.copy()

    if spec.time_column == "date":
        normalised["date"] = pd.to_datetime(normalised["date"], format="%Y-%m-%d", errors="coerce")
        if normalised["date"].isna().any():
            raise DatasetLoadError("dataset 'operations' contains unparseable date values")

    for column in spec.measures:
        if column not in normalised.columns:
            continue
        coerced = pd.to_numeric(normalised[column], errors="coerce")
        if coerced.isna().any():
            raise DatasetLoadError(
                f"dataset '{dataset_id}' column '{column}' contains non-numeric values"
            )
        if (coerced < 0).any():
            logger.warning(
                "negative_measure_clamped",
                extra={
                    "dataset_id": dataset_id.value,
                    "column": column,
                    "rows": int((coerced < 0).sum()),
                },
            )
            coerced = coerced.clip(lower=0)
        normalised[column] = coerced

    return normalised


def load_manifest(data_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the optional dataset manifest. Returns ``None`` when absent."""
    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = (base / "manifest.csv").resolve()
    if base.resolve() not in path.parents or not path.is_file():
        return None
    try:
        return pd.read_csv(path, dtype="string")
    except Exception as exc:  # pragma: no cover - manifest is advisory only
        logger.warning("manifest_unreadable", extra={"error": str(exc)})
        return None


def manifest_row_count(data_dir: Path | None, dataset_id: DatasetId) -> int | None:
    """Row count declared in the manifest for a dataset, if available."""
    manifest = load_manifest(data_dir)
    if manifest is None:
        return None
    matches = manifest.loc[manifest["dataset_id"] == dataset_id.value, "row_count"]
    if matches.empty:
        return None
    try:
        return int(str(matches.iloc[0]))
    except (TypeError, ValueError):  # pragma: no cover - malformed manifest
        return None


def load_all_datasets(data_dir: Path | None = None) -> dict[DatasetId, pd.DataFrame]:
    """Load every registered dataset. Missing optional datasets are skipped."""
    loaded: dict[DatasetId, pd.DataFrame] = {}
    for dataset_id in DATASET_REGISTRY:
        try:
            loaded[dataset_id] = load_dataset(dataset_id, data_dir)
        except DatasetNotFoundError:
            logger.warning("dataset_missing", extra={"dataset_id": dataset_id.value})
    return loaded


def available_dataset_ids(data_dir: Path | None = None) -> list[DatasetId]:
    """Registered dataset ids whose files are present on disk."""
    present: list[DatasetId] = []
    for dataset_id in DATASET_REGISTRY:
        try:
            resolve_dataset_path(dataset_id, data_dir)
        except (DatasetNotFoundError, ConfigurationError):
            continue
        present.append(dataset_id)
    return present


def dataset_spec(dataset_id: DatasetId) -> DatasetSpec:
    return DATASET_REGISTRY[dataset_id]


def data_dir_from_settings(settings: Settings) -> Path:
    return Path(settings.data_dir)


@functools.lru_cache(maxsize=8)
def cached_dataset(dataset_id: DatasetId, data_dir: str) -> pd.DataFrame:
    """Memoised dataset load keyed by (dataset id, data directory).

    Caching returns the *same* DataFrame object, so callers must treat it as
    read-only. Analysis code always takes a filtered copy.
    """
    frame = load_dataset(dataset_id, Path(data_dir))
    frame.flags.writeable = False
    return frame
