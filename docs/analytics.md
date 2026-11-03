# Analytics

## Supported Analysis Types

### kpi_summary
Headline operational metrics: throughput, cycle time, SLA compliance, backlog and
delay rate for a period. Always includes a comparison with the previous period
when available.

### trend_analysis
Monthly trend of one metric with direction, steepness and fit quality. Requires
at least 4 monthly observations. Direction is confirmed against the first/last
change so that a mid-period spike followed by a return is not reported as growth.

### period_comparison
"What changed" between two periods. Routes to the analyzer that owns the
requested metric (cycle time, SLA, backlog, etc.) and returns the comparison
with decomposition (e.g., cycle time movement attributed to queue vs processing).

### cycle_time_analysis
Weighted mean cycle time with a queue/processing decomposition. Reports which
component drove the change and the shift in queue share percentage.

### throughput_analysis
Completed operations per day overall and by dimension. Ranks segments by
throughput and identifies the largest movement between periods.

### backlog_analysis
Closing backlog level, period-over-period change, and accumulation trend. Uses
the last day of the period (not the average) because the average hides a growing
queue.

### sla_analysis
SLA compliance and breach rate recomputed from case counts (never trusted from
the stored percentage). Identifies failing processes and compliance movements.

### bottleneck_analysis
Rank departments or processes by an explainable constraint score combining cycle
time (0.45), queue share (0.35) and backlog pressure (0.20). Each component is
min-max normalised before weighting so no single signal with a large natural
range dominates.

### vendor_analysis
Order-level and operational reliability per vendor. Combines on-time delivery
rate (from orders.csv) with operational delay rate (from operations.csv) to
distinguish transit problems from handling problems.

### department_analysis
Per-department scorecard with target-relative comparison. Cycle time ratio
compares each department against its own SLA target rather than against other
departments, because targets differ by process.

### anomaly_analysis
Detects unusual monthly observations using two complementary detectors: robust
z-score (median/MAD-based) and trailing baseline deviation. Clusters anomalies
by dimension to distinguish one-off events from structural issues.

### process_analysis
Process-level cycle time analysis. Same decomposition as cycle_time_analysis but
filtered to a specific department's processes.

### contribution_analysis
Decomposes a metric change across a dimension (department, process, vendor) to
show which segments drove the overall movement. Supports 11 metrics with custom
value functions. For each segment, the contribution is the segment's change
multiplied by its share of total volume, so a small segment with a large change
can be distinguished from a large segment with a small change.

## Adding a New Analysis Type

1. Add the enum member in `app/models/enums.py`
2. Create the analyzer in `app/analytics/` (or extend an existing one)
3. Register it in `app/analytics/registry.py`
4. Update the default metric mapping in `app/services/analysis_service.py`
5. Add tests
