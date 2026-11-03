import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, askQuestion } from '../services/api';
import type {
  AnalysisPlan,
  EvidenceItem,
  KPIMetric,
  QuestionResponse,
} from '../types';

const EXAMPLE_QUESTIONS = [
  "What is the current SLA compliance rate?",
  "Which department has the highest backlog?",
  "Why did order fulfilment time increase in Q3?",
  "Which vendor has the highest delay rate?",
  "Which process has the largest bottleneck?",
  "Are there unusual operational patterns this quarter?",
  "Which operational metrics are deteriorating?",
  "Where should management investigate first?",
];

export default function AskPage() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<QuestionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const navigate = useNavigate();

  const handleSubmit = async (q?: string) => {
    const text = q || question;
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askQuestion(text);
      setResult(data);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e), 'unknown', 0));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <h2>Ask Operations Intelligence</h2>
        <p>Ask a question about operational performance, bottlenecks, or anomalies</p>
      </div>

      <div className="card">
        <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
          <textarea
            className="input"
            placeholder="e.g. Why did order fulfilment time increase in Q3?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            maxLength={600}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', alignItems: 'center' }}>
            <div className="example-questions">
              {EXAMPLE_QUESTIONS.slice(0, 4).map((q) => (
                <span
                  key={q}
                  className="example-question"
                  onClick={() => { setQuestion(q); handleSubmit(q); }}
                >
                  {q}
                </span>
              ))}
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading || !question.trim()}>
              {loading ? 'Analysing…' : 'Run Analysis'}
            </button>
          </div>
        </form>
      </div>

      {loading && <div className="loading">Running deterministic analysis…</div>}
      {error && <RejectionNotice error={error} />}

      {result && <InsightDisplay result={result} onOpenDetail={navigate} />}

      {!result && !loading && !error && (
        <div className="card" style={{ marginTop: '24px' }}>
          <div className="card-header">Additional Example Questions</div>
          <div className="example-questions">
            {EXAMPLE_QUESTIONS.slice(4).map((q) => (
              <span
                key={q}
                className="example-question"
                onClick={() => { setQuestion(q); handleSubmit(q); }}
              >
                {q}
              </span>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

/**
 * The question could not be answered, and the backend said why.
 *
 * Refusal is a designed outcome, not an error: showing the reason tells the user
 * what this platform can and cannot do, which a bare "failed" cannot.
 */
function RejectionNotice({ error }: { error: ApiError }) {
  return (
    <div className="card" style={{ marginTop: '24px', borderLeft: '3px solid var(--color-severity-medium)' }}>
      <div className="card-header">This question cannot be answered</div>
      <p style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--color-text-secondary)' }}>
        {error.message}
      </p>
      <p style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginTop: '8px' }}>
        {error.code} · HTTP {error.status || 'n/a'}
      </p>
    </div>
  );
}

function InsightDisplay({
  result,
  onOpenDetail,
}: {
  result: QuestionResponse;
  onOpenDetail: (path: string) => void;
}) {
  return (
    <>
      <div className="card" style={{ marginTop: '24px' }}>
        <div className="card-header">Answer</div>
        <p style={{ fontSize: '14px', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
          {result.answer}
        </p>
        <Provenance result={result} />
      </div>

      <PlanCard plan={result.plan} />

      {result.kpis.length > 0 && (
        <>
          <div className="card-header" style={{ marginTop: '24px' }}>Measured KPIs</div>
          <p className="section-note">
            Computed in Python from the source datasets. These figures are authoritative.
          </p>
          <div className="kpi-grid">
            {result.kpis.map((kpi, i) => <KpiTile key={i} kpi={kpi} />)}
          </div>
        </>
      )}

      {result.findings.length > 0 && (
        <div className="card">
          <div className="card-header">Deterministic Findings ({result.findings.length})</div>
          {result.findings.map((f, i) => (
            <div key={i} style={{ marginBottom: '14px', paddingLeft: '8px' }}>
              <SeverityBadge severity={f.severity} />
              <strong style={{ marginLeft: '8px' }}>{f.title}</strong>
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
                {f.detail}
              </p>
              {f.statistic && (
                <p style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginTop: '4px' }}>
                  {f.statistic}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {result.key_findings.length > 0 && (
        <div className="card">
          <div className="card-header">Key Findings</div>
          {result.key_findings.map((f, i) => (
            <div key={i} style={{ marginBottom: '12px', paddingLeft: '8px' }}>
              <SeverityBadge severity={f.severity} />
              <span style={{ marginLeft: '8px', fontSize: '13px' }}>{f.statement}</span>
            </div>
          ))}
        </div>
      )}

      {result.recommendations.length > 0 && (
        <div className="card">
          <div className="card-header">Recommendations</div>
          {result.recommendations.map((r, i) => (
            <div key={i} style={{ marginBottom: '12px' }}>
              <SeverityBadge severity={r.priority} />
              <strong style={{ marginLeft: '8px', fontSize: '13px' }}>{r.action}</strong>
              <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px', paddingLeft: '8px' }}>
                {r.rationale}
              </p>
            </div>
          ))}
        </div>
      )}

      {result.evidence.length > 0 && (
        <div className="card">
          <div className="card-header">Evidence ({result.evidence.length} items)</div>
          <p className="section-note">
            The narrative above may only restate these figures. Each one records the
            calculation that produced it.
          </p>
          <div className="evidence-list">
            {result.evidence.map((e, i) => <EvidenceTile key={i} item={e} />)}
          </div>
        </div>
      )}

      {result.caveats.length > 0 && (
        <div className="card">
          <div className="card-header">Caveats</div>
          {result.caveats.map((c, i) => (
            <div key={i} className="caveat">{c}</div>
          ))}
        </div>
      )}

      <div style={{ marginTop: '16px' }}>
        <button className="btn btn-secondary" onClick={() => onOpenDetail(`/analysis/${result.analysis_id}`)}>
          View full analysis detail
        </button>
      </div>
    </>
  );
}

/**
 * Where the answer came from and whether its numbers were checked.
 *
 * Showing this matters: `deterministic_fallback` means no LLM was configured, and
 * an ungrounded number would mean the narrative was replaced before it shipped.
 */
function Provenance({ result }: { result: QuestionResponse }) {
  const { grounding } = result;
  return (
    <div style={{ marginTop: '12px', fontSize: '11px', color: 'var(--color-text-muted)' }}>
      <div>
        Generated by:{' '}
        {result.generated_by === 'llm'
          ? `LLM explanation${result.model ? ` (${result.model})` : ''}, validated against the evidence`
          : 'deterministic fallback (no LLM configured)'}
      </div>
      <div style={{ marginTop: '4px' }}>
        Grounding: {grounding.matched}/{grounding.numbers_checked} figures matched
        {grounding.matched_after_rounding > 0 &&
          ` (${grounding.matched_after_rounding} after rounding)`}
        {grounding.unsupported > 0 && ` · ${grounding.unsupported} unsupported`}
        {grounding.grounded ? ' · verified' : ' · narrative replaced'}
      </div>
      <div style={{ marginTop: '4px' }}>
        Plan: {result.plan.analysis_type} · {result.plan.metric}
        {result.plan.period ? ` · ${result.plan.period}` : ''} · goal {result.plan.reasoning_goal} ·{' '}
        {result.row_count_examined.toLocaleString()} rows examined in {result.duration_ms.toFixed(0)}ms
      </div>
    </div>
  );
}

/** The validated plan, so a reader can see what actually ran. */
function PlanCard({ plan }: { plan: AnalysisPlan }) {
  return (
    <div className="card">
      <div className="card-header">Analysis Plan</div>
      <table className="data-table">
        <tbody>
          <tr><th style={{ width: '160px' }}>Analysis</th><td>{label(plan.analysis_type)}</td></tr>
          <tr><th>Metric</th><td>{label(plan.metric)}</td></tr>
          <tr><th>Period</th><td>{plan.period ?? 'All available'}</td></tr>
          <tr><th>Comparison</th><td>{plan.comparison_period ?? 'None'}</td></tr>
          <tr><th>Reasoning goal</th><td>{label(plan.reasoning_goal)}</td></tr>
          <tr><th>Datasets</th><td>{plan.required_datasets.join(', ')}</td></tr>
          {plan.dimensions.length > 0 && <tr><th>Dimensions</th><td>{plan.dimensions.join(', ')}</td></tr>}
          {plan.department && <tr><th>Department</th><td>{plan.department}</td></tr>}
          {plan.process && <tr><th>Process</th><td>{plan.process}</td></tr>}
          {plan.vendor_id && <tr><th>Vendor</th><td>{plan.vendor_id}</td></tr>}
        </tbody>
      </table>
      {plan.notes && <div className="caveat" style={{ marginTop: '8px' }}>{plan.notes}</div>}
    </div>
  );
}

function KpiTile({ kpi }: { kpi: KPIMetric }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{kpi.name}</div>
      <div className="kpi-value">
        {formatValue(kpi.value, kpi.unit)}
        {kpi.unit !== 'percent' && kpi.unit !== 'count' && (
          <span className="kpi-unit">{kpi.unit}</span>
        )}
      </div>
      <div className="kpi-period">
        {kpi.period ?? 'all periods'} · {kpi.source_dataset}
      </div>
    </div>
  );
}

function EvidenceTile({ item }: { item: EvidenceItem }) {
  const changeClass =
    item.percentage_change && item.percentage_change > 0
      ? 'positive'
      : item.percentage_change && item.percentage_change < 0
        ? 'negative'
        : '';
  return (
    <div className={`evidence-item severity-${item.severity}`}>
      <div className="evidence-metric">{item.metric}</div>
      <div>
        <span className="evidence-value">{formatValue(item.value, item.unit)}</span>
        {item.percentage_change !== null && (
          <span className={`evidence-change ${changeClass}`}>
            {item.percentage_change > 0 ? '+' : ''}{item.percentage_change.toFixed(1)}%
          </span>
        )}
      </div>
      {item.comparison_value !== null && (
        <div className="kpi-period">
          Previous period: {formatValue(item.comparison_value, item.unit)}
          {item.comparison_period ? ` (${item.comparison_period})` : ''}
        </div>
      )}
      {item.dimension && <div className="evidence-desc">By {item.dimension}</div>}
      <div className="evidence-desc">{item.calculation_description}</div>
    </div>
  );
}

/** `55.38`, `97.57%`, `1,234`, or `unavailable` for an undefined figure. */
function formatValue(value: number | null, unit: string): string {
  if (value === null) return 'unavailable';
  if (unit === 'percent') return `${value.toFixed(2)}%`;
  if (unit === 'count' || Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function label(value: string): string {
  return value.replace(/_/g, ' ');
}

function SeverityBadge({ severity }: { severity: string }) {
  const cls = severity ? `badge badge-${severity.toLowerCase()}` : 'badge badge-info';
  return <span className={cls}>{severity || 'info'}</span>;
}
