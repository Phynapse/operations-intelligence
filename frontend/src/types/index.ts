export interface KPIMetric {
  name: string;
  value: number | null;
  unit: string;
  period: string | null;
  dimension: string | null;
  source_dataset: string;
  calculation: string;
}

export interface EvidenceItem {
  metric: string;
  value: number | null;
  comparison_value: number | null;
  absolute_change: number | null;
  percentage_change: number | null;
  unit: string;
  dimension: string | null;
  period: string | null;
  comparison_period: string | null;
  source_dataset: string;
  calculation_description: string;
  severity: string;
  sample_size: number | null;
  is_estimate: boolean;
}

export interface Finding {
  title: string;
  detail: string;
  severity: string;
  metric: string | null;
  dimension: string | null;
  period: string | null;
  supporting_evidence: string[];
  is_statistical: boolean;
  statistic: string | null;
}

export interface AnalysisPlan {
  analysis_type: string;
  metric: string;
  dimensions: string[];
  period: string | null;
  comparison_period: string | null;
  department: string | null;
  process: string | null;
  vendor_id: string | null;
  top_n: number;
  required_datasets: string[];
  reasoning_goal: string;
  notes: string | null;
}

export interface AnalysisResult {
  analysis_id: string;
  created_at: string;
  question: string | null;
  plan: AnalysisPlan;
  evidence: EvidenceItem[];
  kpis: KPIMetric[];
  findings: Finding[];
  caveats: string[];
  row_count_examined: number;
  duration_ms: number;
  status: string;
}

export interface KeyFinding {
  statement: string;
  severity: string;
  evidence_ids: string[];
  metric: string | null;
}

export interface Recommendation {
  action: string;
  rationale: string;
  evidence_ids: string[];
  priority: string;
  expected_metric_effect: string | null;
}

export interface OperationalInsight {
  answer: string;
  key_findings: KeyFinding[];
  recommendations: Recommendation[];
  caveats: string[];
  generated_by: string;
  model: string | null;
  unsupported_claims_detected: boolean;
}

/** What the evidence validator concluded about the generated narrative. */
export interface GroundingSummary {
  numbers_checked: number;
  matched: number;
  matched_after_rounding: number;
  unsupported: number;
  grounded: boolean;
  issues: string[];
}

/**
 * Response envelope for `POST /questions`.
 *
 * The deterministic half (`plan`, `kpis`, `findings`, `evidence`) is
 * authoritative; `answer` and the structured sections below may only restate it.
 */
export interface QuestionResponse {
  analysis_id: string;
  question: string;
  plan: AnalysisPlan;
  kpis: KPIMetric[];
  findings: Finding[];
  evidence: EvidenceItem[];
  answer: string;
  key_findings: KeyFinding[];
  recommendations: Recommendation[];
  caveats: string[];
  reasoning_goal: string;
  generated_by: 'llm' | 'deterministic_fallback';
  model: string | null;
  unsupported_claims_detected: boolean;
  grounding: GroundingSummary;
  created_at: string;
  row_count_examined: number;
  duration_ms: number;
  status: string;
}

export interface DatasetInfo {
  dataset_id: string;
  file_name: string;
  description: string;
  grain: string;
  primary_key: string;
  time_column: string | null;
  dimensions: string[];
  measures: string[];
  row_count: number;
  column_count: number;
  loaded: boolean;
}

export interface DatasetSummary {
  dataset_id: string;
  row_count: number;
  column_count: number;
  columns: string[];
  row_count_matches_manifest: boolean;
  period_start: string | null;
  period_end: string | null;
  distinct_departments: number;
  distinct_processes: number;
  distinct_vendors: number;
  numeric_measures: Record<string, { mean: number | null; min: number | null; max: number | null }>;
  completeness: Record<string, number>;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
  datasets_loaded: number;
  datasets_available: number;
  llm_configured: boolean;
}

// -- Dashboard types --

export interface DashboardKPI {
  name: string;
  value: number | null;
  unit: string;
  period: string | null;
}

export interface TrendPoint {
  period: string;
  value: number | null;
}

export interface BottleneckEntry {
  name: string;
  cycle_time: number | null;
  queue_share: number | null;
  backlog: number | null;
}

export interface DepartmentOverview {
  department: string;
  throughput: number | null;
  cycle_time: number | null;
  sla_compliance: number | null;
  backlog: number | null;
}

export interface DashboardData {
  generated_at: string;
  period_start: string | null;
  period_end: string | null;
  kpis: DashboardKPI[];
  throughput_trend: TrendPoint[];
  cycle_time_trend: TrendPoint[];
  sla_trend: TrendPoint[];
  backlog_trend: TrendPoint[];
  bottlenecks: BottleneckEntry[];
  departments: DepartmentOverview[];
}
