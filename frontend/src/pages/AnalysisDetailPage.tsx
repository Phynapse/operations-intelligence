import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { fetchAnalysis } from '../services/api';
import type { AnalysisResult, KPIMetric, EvidenceItem } from '../types';

export default function AnalysisDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    // Capturing the narrowed id keeps it a `string` inside the closure,
    // where the effect body may run again after the guard has passed.
    const analysisId = id;
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchAnalysis(analysisId);
        if (!cancelled) setResult(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [id]);

  if (loading) return <div className="loading">Loading analysis…</div>;
  if (error) return <div className="loading" style={{ color: 'var(--color-severity-critical)' }}>Error: {error}</div>;
  if (!result) return <div className="loading">No analysis found</div>;

  const plan = result.plan;

  return (
    <>
      <div className="page-header">
        <h2>Analysis Detail</h2>
        <p>{result.question || `${plan.analysis_type.replace(/_/g, ' ')} — ${plan.metric.replace(/_/g, ' ')}`}</p>
      </div>

      <div className="card">
        <div className="card-header">Analysis Plan</div>
        <table className="data-table">
          <tbody>
            <tr><th style={{ width: '160px' }}>Analysis Type</th><td>{plan.analysis_type.replace(/_/g, ' ')}</td></tr>
            <tr><th>Metric</th><td>{plan.metric.replace(/_/g, ' ')}</td></tr>
            <tr><th>Period</th><td>{plan.period || 'All periods'}</td></tr>
            {plan.comparison_period && <tr><th>Comparison</th><td>{plan.comparison_period}</td></tr>}
            {plan.department && <tr><th>Department</th><td>{plan.department}</td></tr>}
            {plan.process && <tr><th>Process</th><td>{plan.process}</td></tr>}
            {plan.dimensions.length > 0 && <tr><th>Dimensions</th><td>{plan.dimensions.join(', ')}</td></tr>}
            <tr><th>Reasoning Goal</th><td>{plan.reasoning_goal.replace(/_/g, ' ')}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="kpi-grid" style={{ marginTop: '24px' }}>
        {result.kpis.map((kpi, i) => (
          <KpiDisplay key={i} kpi={kpi} />
        ))}
      </div>

      {result.evidence.length > 0 && (
        <div className="card">
          <div className="card-header">Evidence ({result.evidence.length} items)</div>
          <div className="evidence-list">
            {result.evidence.map((ev, i) => (
              <EvidenceDisplay key={i} item={ev} />
            ))}
          </div>
        </div>
      )}

      {result.findings.length > 0 && (
        <div className="card">
          <div className="card-header">Findings ({result.findings.length})</div>
          {result.findings.map((f, i) => (
            <div key={i} style={{ marginBottom: '12px', paddingLeft: '8px' }}>
              <SeverityBadge severity={f.severity} />
              <strong style={{ marginLeft: '8px' }}>{f.title}</strong>
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
                {f.detail}
              </p>
            </div>
          ))}
        </div>
      )}

      {result.caveats.length > 0 && (
        <div className="card">
          <div className="card-header">Notes</div>
          {result.caveats.map((c, i) => (
            <div key={i} className="caveat">{c}</div>
          ))}
        </div>
      )}

      <div className="card" style={{ marginTop: '16px' }}>
        <div className="card-header">Metadata</div>
        <p style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>
          ID: {result.analysis_id} &middot; {result.row_count_examined.toLocaleString()} records &middot; {result.duration_ms.toFixed(1)}ms &middot; {new Date(result.created_at).toLocaleString()}
        </p>
      </div>
    </>
  );
}

function KpiDisplay({ kpi }: { kpi: KPIMetric }) {
  const valueStr = kpi.value !== null && kpi.value !== undefined
    ? (Math.abs(kpi.value) >= 1000 ? kpi.value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : kpi.value.toFixed(2))
    : '—';
  const unitStr = kpi.unit === 'percent' ? '%' : kpi.unit === 'currency' ? '$' : ` ${kpi.unit}`;
  return (
    <div className="kpi-card">
      <div className="kpi-label">{kpi.name}</div>
      <div className="kpi-value">{valueStr}<span className="kpi-unit">{unitStr}</span></div>
      {kpi.period && <div className="kpi-period">{kpi.period}</div>}
    </div>
  );
}

function EvidenceDisplay({ item }: { item: EvidenceItem }) {
  const changeText = item.percentage_change !== null && item.percentage_change !== undefined
    ? `${item.percentage_change >= 0 ? '+' : ''}${item.percentage_change.toFixed(1)}%`
    : '';
  const changeClass = item.percentage_change && item.percentage_change > 0 ? 'positive' : item.percentage_change && item.percentage_change < 0 ? 'negative' : '';

  return (
    <div className={`evidence-item severity-${item.severity}`}>
      <div className="evidence-metric">{item.metric}</div>
      <div>
        <span className="evidence-value">{item.value !== null ? item.value.toFixed(2) : '—'} {item.unit}</span>
        {changeText && <span className={`evidence-change ${changeClass}`}>{changeText}</span>}
      </div>
      {item.comparison_value !== null && item.comparison_value !== undefined && (
        <div style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>
          Previous: {item.comparison_value.toFixed(2)} {item.unit}
        </div>
      )}
      {item.dimension && (
        <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>
          Dimension: {item.dimension}
        </div>
      )}
      <div className="evidence-desc">{item.calculation_description}</div>
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const cls = severity ? `badge badge-${severity.toLowerCase()}` : 'badge badge-info';
  return <span className={cls}>{severity || 'info'}</span>;
}
