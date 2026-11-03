"""Dataset catalogue routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.data.repository import get_repository
from app.schemas.operational import DatasetInfo, DatasetSummary

router = APIRouter()


@router.get("", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    repository = get_repository()
    return repository.catalogue()


@router.get("/{dataset_id}", response_model=DatasetSummary)
async def get_dataset_summary(dataset_id: str) -> DatasetSummary:
    from app.data.loader import parse_dataset_id

    resolved = parse_dataset_id(dataset_id)
    repository = get_repository()
    frame = repository.frame(resolved)
    manifest_count = repository.declared_row_count(resolved)
    numeric_cols = [c for c in frame.columns if frame[c].dtype in ("float64", "Int64", "int64")]

    return DatasetSummary(
        dataset_id=resolved,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=tuple(frame.columns),
        row_count_matches_manifest=(manifest_count is None or manifest_count == len(frame)),
        distinct_departments=int(frame["department"].nunique())
        if "department" in frame.columns
        else 0,
        distinct_processes=int(frame["process"].nunique()) if "process" in frame.columns else 0,
        distinct_vendors=int(frame["vendor_id"].dropna().nunique())
        if "vendor_id" in frame.columns
        else 0,
        numeric_measures={
            col: {
                "mean": round(float(frame[col].mean()), 2) if frame[col].notna().any() else None,
                "min": round(float(frame[col].min()), 2) if frame[col].notna().any() else None,
                "max": round(float(frame[col].max()), 2) if frame[col].notna().any() else None,
            }
            for col in numeric_cols[:10]
        },
        completeness={
            col: round(float(frame[col].notna().mean()) * 100, 1) for col in numeric_cols[:10]
        },
    )
