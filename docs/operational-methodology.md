# Operational Methodology

This document explains how the synthetic operational data was generated, what the embedded patterns represent, and how the analytics layer is designed to detect them.

## Data Generation Philosophy

The synthetic dataset is not random. It is produced by an explicit **operating model** that mirrors how real operational systems behave:

```
Daily Arrivals → Service Capacity → Closing Backlog
    ↓
Queue Time (waiting) + Processing Time (working) = Total Cycle Time
    ↓
Promise/Delay Metrics → SLA Compliance
```

Because every derived dataset (`sla_metrics.csv`, `vendors.csv`, `orders.csv`) is aggregated **from** `operations.csv` or driven **by** it, the numbers are internally consistent by construction. Relationships that would be accidental in independent CSVs are guaranteed:

* `total_cycle_time_hours` = `queue_time_hours` + `processing_time_hours` (exact)
* `sla_met_cases` + `sla_breached_cases` = `total_cases` (exact)
* `sla_compliance_percentage` = `sla_met_cases / total_cases * 100` (exact)
* `actual_time_hours` = `fulfillment_time_hours` + `transit_time_hours` (exact)
* `delayed_operations` ≤ `completed_operations` (always)

## Embedded Patterns

Seven realistic operational patterns are embedded. They are designed to be **discoverable but not obvious** — a human analyst or the analytics layer must compute and compare to surface them.

| ID | Signal | Department / Process | Magnitude |
|----|--------|---------------------|-----------|
| P1 | Carrier processing degrades | DEP-102 outbound shipping | +25% over the year |
| P2 | Demand grows without capacity | DEP-105 contract administration | Backlog 0 → 47 items |
| P3 | Vendor delay rate high | VND-102 RapidRoute Freight | 26% vs ~5% peers |
| P4 | Procurement SLA deteriorates | DEP-105 (Q4) | 98% → 26% |
| P5 | Queue time climbs, processing flat | DEP-104 incident resolution | 1.2 → 7.3 hours |
| P6 | Processing improves | DEP-103 invoice processing | -34% over the year |
| P7 | Vendor delay degrades in H2 | VND-310 NetGuard Support | Ramps through the year |

### P1: Carrier Processing Degradation
**Where:** DEP-102 outbound shipping  
**What:** The base processing time drifts upward at ~0.00058/day (~+25% annually). Queue time stays relatively flat because demand is stable.  
**What to find:** Cycle time analysis will show the increase is entirely in processing, not queue. Vendor analysis will identify VND-102 as the dominant contributor.

### P2: Demand Growth Without Capacity
**Where:** DEP-105 contract administration  
**What:** Daily demand slope of 0.014 with no capacity increase. Backlog accumulates from near-zero to ~47 items by year-end.  
**What to find:** Backlog analysis shows steady accumulation. Bottleneck analysis ranks this process high on backlog pressure.

### P3: Persistently High Vendor Delay
**Where:** VND-102 RapidRoute Freight (outbound shipping)  
**What:** Intrinsic delay floor of 26% vs ~3-5% for other logistics vendors.  
**What to find:** Vendor analysis surfaces VND-102 as an outlier on both operational delay rate and on-time delivery rate. Contribution analysis shows VND-102 accounts for most of the shipping cycle time movement.

### P4: Procurement SLA Collapse
**Where:** DEP-105 contract administration (Q4)  
**What:** Combined effect of P2 (growing backlog) and tight SLA target (16h). Compliance drops from ~98% in Q1 to ~26% in Q4.  
**What to find:** SLA analysis identifies the quarter-by-quarter compliance trajectory. Period comparison between Q1 and Q4 shows the largest breach movement.

### P5: Queue Time Explosion
**Where:** DEP-104 incident resolution  
**What:** Arrivals slope of 0.017/day with flat processing time (5.4h). Queue time grows from ~1.2h to ~7.3h.  
**What to find:** Cycle time analysis decomposes the movement: queue contribution rises from ~18% to ~57%. Anomaly detection flags the final quarter as unusual.

### P6: Processing Improvement
**Where:** DEP-103 invoice processing  
**What:** Base processing time drifts downward at -0.0012/day (~34% improvement). Queue time stable.  
**What to find:** One of the few positive signals. Trend analysis shows negative slope. Department analysis shows DEP-103 improving relative to target.

### P6b: Order Handling Improvement
**Where:** Orders → Standard Supplies category  
**What:** Internal handling time drifts downward at -0.0003/day.  
**What to find:** Fulfillment time analysis on orders.csv shows improvement for this category.

### P7: Vendor Degradation in H2
**Where:** VND-310 NetGuard Support (DEP-104 service request fulfilment)  
**What:** Delay floor ramps at 0.00052/day — effectively a vendor whose reliability deteriorates through the year.  
**What to find:** Vendor trend analysis shows the ramp. Contribution analysis for DEP-104 cycle time change identifies VND-310 as a growing share.

## Analytics Design to Detect Patterns

### Deterministic Calculations
Every metric uses pure Python functions (`app/utils/calculations.py`) with strict safety:
* Zero denominators → `None`, never NaN or infinity
* Negative values where inappropriate → clamped to zero
* Missing data → excluded, not imputed

### Analyzers
Each analyzer is a pure function of a filtered DataFrame and a validated plan:

| Analyzer | Primary Pattern Detected |
|----------|-------------------------|
| `KPIAnalyzer` | Headline overview (P1-P7 all contribute) |
| `TrendAnalyzer` | P1, P5, P6, P7 (monthly slope + R²) |
| `PeriodComparisonAnalyzer` | P2, P4 (Q1 vs Q4 decomposition) |
| `CycleTimeAnalyzer` | P1, P5, P6 (queue vs processing decomposition) |
| `ThroughputAnalyzer` | P2, P6 (completed operations/day) |
| `BacklogAnalyzer` | P2, P4 (closing backlog + change) |
| `SLAAnalyzer` | P4 (compliance recomputed from counts) |
| `BottleneckAnalyzer` | P1, P2, P4, P5 (weighted constraint score) |
| `VendorAnalyzer` | P3, P7 (delay rate + on-time delivery) |
| `DepartmentAnalyzer` | P1-P7 (per-department scorecards) |
| `AnomalyAnalyzer` | P5, P7 (z-score + baseline deviation) |
| `ContributionAnalyzer` | P1, P3, P7 (dimension-level decomposition) |

### Evidence-First Explanation
The AI layer receives **only** structured evidence items:
* Metric name, value, comparison value, absolute/percentage change
* Unit, dimension, period, source dataset
* Human-readable calculation description

The LLM never computes. It only arranges and phrases the numbers Python produced.

## Regenerating the Dataset

```bash
python scripts/generate_synthetic_data.py --out data --seed 20250101
```

The output is byte-stable for a fixed seed. The generator version and seed are recorded in `manifest.csv`.

## Extending the Patterns

To add a new pattern:
1. Modify the `PROCESSES` or `VENDORS` list in `scripts/generate_synthetic_data.py`
2. Adjust the relevant `drift` or `slope` parameters
3. Re-run the generator
4. The analytics layer will automatically detect the new signal — no code changes required.

This separation (data patterns in the generator, detection in the analytics) is the core of the platform's maintainability.