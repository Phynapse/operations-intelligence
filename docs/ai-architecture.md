# AI Architecture

## Deterministic Calculation

Every metric is computed by a pure Python function that takes validated inputs
and returns structured outputs. There is no randomness, no LLM involvement, and
no external state.

### Core Formulas

| Metric | Formula |
|--------|---------|
| Percentage change | `(current - previous) / previous * 100` |
| SLA compliance | `sla_met_cases / total_cases * 100` |
| SLA breach rate | `sla_breached_cases / total_cases * 100` |
| Average cycle time | weighted mean of `total_cycle_time_hours` by `completed_operations` |
| Queue contribution | `queue_time / total_cycle_time * 100` |
| Processing contribution | `processing_time / total_cycle_time * 100` |
| Backlog change | `current_backlog - previous_backlog` |
| Throughput | `completed_operations / period_days` |
| Delay rate | `delayed_operations / total_operations * 100` |

### Safety

* Zero denominators return `None`, not NaN or infinity.
* Negative values where inappropriate are clamped to zero.
* Missing data is excluded, not imputed.
* Invalid values raise validation errors at the data boundary.

### Statistical Methods

* **Z-scores** use the population formula (divide by N, not N-1).
* **Linear trends** use least-squares regression with R²; flat series report slope=0 with R²=None.
* **Anomaly detection** uses two complementary detectors:
  * Modified z-score (median/MAD-based, threshold ±3.5)
  * Trailing baseline deviation (3-month mean, threshold ±15%)
* **Bottleneck scoring** uses min-max normalisation across the compared set, weighted:
  * cycle time: 0.45
  * queue share: 0.35
  * backlog pressure: 0.20

## LLM Role

The LLM **interprets** evidence; it does not **compute** it.

### What the LLM receives:
* The user's question
* A capped, severity-ordered list of evidence items
* Each item: metric name, value, comparison value, change, unit, dimension, period, source dataset, calculation description
* Deterministic finding statements
* Caveats about data quality and limitations

### What the LLM produces:
* A narrative answer (max 2000 chars)
* Structured key findings with severity
* Evidence-grounded recommendations
* It must NOT introduce new numbers not in the evidence package

## AI Architecture

### QuestionInterpreter (Phase 1: deterministic)
* Keyword-based routing to analysis type and metric
* Period detection from question text (Q1, 2025-03, etc.)
* Dimension extraction (department, process, vendor, etc.)
* Reasoning goal inference (explain_change, identify_bottleneck, etc.)
* Phase 2: replace with LLM call using same input/output contract

### AnalysisPlanner
* Maps intent to controlled enums (AnalysisType, Metric, Dimension)
* Resolves periods against available data
* Infers comparison window when needed
* Validates filter values against known datasets
* Returns a fully validated AnalysisPlan

### ExplanationGenerator (Phase 1: deterministic)
* Assembles answer from analyzer findings
* Orders evidence by severity and movement magnitude
* Generates placeholder recommendations
* Phase 2: replace with LLM call using evidence package as prompt

### EvidenceService
* Receives the raw `AnalysisOutcome` from the analytics layer
* Orders evidence items by severity (critical → info), then by absolute change magnitude
* Caps the package at 24 items to stay within LLM context limits
* Produces an `EvidencePackage` that the explanation generator consumes
* The evidence package is the only data the LLM sees — it never receives raw DataFrames

### EvidenceValidator
* Extracts every number from the LLM-generated narrative
* Checks each number against the evidence package index
* Accepts exact matches and rounding-within-tolerance (±0.5 of the last decimal)
* If any number is unsupported, the narrative is replaced by the deterministic fallback
* Reports a `GroundingSummary` so the frontend can show whether grounding passed

### LLMProvider
* Abstract interface for any OpenAI-compatible endpoint
* OpenAI-compatible implementation (works with OpenAI, Azure, Ollama, etc.)
* FakeProvider for testing (no API key required)
