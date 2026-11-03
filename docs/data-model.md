# Data Model

## Datasets

Five CSV files with deterministic, internally consistent synthetic data.

### operations.csv

| Column | Type | Description |
|--------|------|-------------|
| operation_id | string | Unique identifier per department-process-day |
| date | date | Calendar date |
| period | YYYY-MM | Month bucket |
| quarter | YYYY-Qn | Quarter bucket |
| department | string | Department id (DEP-101…DEP-105) |
| process | string | Process name |
| operation_type | string | Work category |
| status | enum | completed / at_risk / delayed |
| priority | enum | standard / expedited / critical |
| completed_operations | int | Work completed that day |
| delayed_operations | int | Late relative to internal promise |
| processing_time_hours | float | Actual work hours (per operation average) |
| queue_time_hours | float | Time spent waiting |
| total_cycle_time_hours | float | queue_time + processing_time (always exact) |
| employee_count | int | Allocated headcount |
| vendor_id | string/null | External vendor, or null for internal |
| backlog_open_items | float | End-of-day open items |

### orders.csv

| Column | Type | Description |
|--------|------|-------------|
| order_id | string | Unique identifier |
| order_date | date | When the order was placed |
| period / quarter | string | Time buckets |
| department | string | Originating department |
| product_category | string | Supplies / Equipment / Furniture / Digital / Safety |
| order_value | float | Monetary value |
| promised_time_hours | float | Customer-facing delivery promise |
| actual_time_hours | float | Realised end-to-end time |
| fulfillment_time_hours | float | Our own handling time |
| transit_time_hours | float | Carrier transit time |
| status | enum | delivered / delivered_late / returned / cancelled |
| vendor_id | string | Freight carrier |
| delay_hours | float | Hours late (zero if on time or cancelled) |

### sla_metrics.csv

Monthly rollup of SLA performance per department-process.

| Column | Type | Description |
|--------|------|-------------|
| department | string | Department name (e.g. "Fulfillment") |
| process | string | Process name |
| period | YYYY-MM | Month bucket |
| quarter | YYYY-Qn | Quarter bucket |
| total_cases | int | Total operations in the period |
| sla_target_hours | float | SLA target cycle time |
| average_cycle_time_hours | float | Observed average cycle time |
| average_queue_time_hours | float | Observed average queue time |
| sla_met_cases | int | Cases delivered within SLA target |
| sla_breached_cases | int | Cases that breached SLA target |
| sla_compliance_percentage | float | sla_met_cases / total_cases × 100 (exact) |
| backlog_open_items | float | End-of-period open items |

### vendors.csv

One row per vendor with aggregated reliability metrics.

| Column | Type | Description |
|--------|------|-------------|
| vendor_id | string | Unique vendor identifier (e.g. VND-102) |
| vendor_name | string | Human-readable vendor name |
| category | string | Vendor category (e.g. Logistics, IT, Consulting) |
| department | string | Primary associated department |
| monthly_volume | int | Typical monthly transaction volume |
| average_processing_time | float | Average processing time in hours |
| delay_rate | float | Percentage of delayed operations |
| sla_compliance | float | SLA compliance percentage |

### departments.csv

Five departments with id, name, operational area and headcount.

| Column | Type | Description |
|--------|------|-------------|
| department_id | string | Unique department identifier (DEP-101…DEP-105) |
| department_name | string | Human-readable name |
| operational_area | string | Functional area (e.g. Front Office, Supply Chain) |
| headcount | int | Allocated staff count |

## Consistency Guarantees

* `total_cycle_time_hours` = `queue_time_hours` + `processing_time_hours` (exact)
* `sla_met_cases` + `sla_breached_cases` = `total_cases` (exact)
* `sla_compliance_percentage` = `sla_met_cases / total_cases * 100` (exact)
* `actual_time_hours` = `fulfillment_time_hours` + `transit_time_hours` (exact)
* `delay_hours` = `actual_time_hours - promised_time_hours` for late orders, zero otherwise
* Cancelled orders have zero `delay_hours`
* `delayed_operations` ≤ `completed_operations` (always)
* All percentages ≥ 0 and ≤ 100

## Embedded Patterns

Seven deliberate but realistic operational patterns are embedded:

| ID | Signal | Where | Magnitude |
|----|--------|-------|-----------|
| P1 | Carrier processing degrades | outbound shipping | +25% over the year |
| P2 | Demand grows without capacity | contract administration | backlog 0 → 47 items |
| P3 | Vendor delay rate high | VND-102 RapidRoute | 26% vs ~5% peers |
| P4 | Procurement SLA deteriorates | Q4 | 98% → 26% |
| P5 | Queue time climbs, processing flat | incident resolution | 1.2 → 7.3 hours |
| P6 | Processing improves | invoice processing | -34% over the year |
| P7 | Vendor delay degrades in H2 | VND-310 NetGuard | ramps through the year |
