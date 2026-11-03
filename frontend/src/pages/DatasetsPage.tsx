import { useState, useEffect } from 'react';
import { fetchDatasets, fetchDatasetSummary } from '../services/api';
import type { DatasetInfo, DatasetSummary } from '../types';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedSummary, setSelectedSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const data = await fetchDatasets();
        if (!cancelled) setDatasets(data);
      } catch (e) {
        // ignore
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const handleSelect = async (datasetId: string) => {
    try {
      const summary = await fetchDatasetSummary(datasetId);
      setSelectedSummary(summary);
    } catch {
      setSelectedSummary(null);
    }
  };

  if (loading) return <div className="loading">Loading datasets…</div>;

  return (
    <>
      <div className="page-header">
        <h2>Datasets</h2>
        <p>Synthetic operational data powering the analysis engine</p>
      </div>

      <div className="kpi-grid">
        {datasets.map((ds) => (
          <div
            key={ds.dataset_id}
            className="kpi-card"
            style={{ cursor: 'pointer' }}
            onClick={() => handleSelect(ds.dataset_id)}
          >
            <div className="kpi-label">{ds.dataset_id}</div>
            <div className="kpi-value" style={{ fontSize: '18px' }}>{ds.row_count.toLocaleString()} <span className="kpi-unit">rows</span></div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>{ds.description}</div>
            <div style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginTop: '4px' }}>{ds.grain}</div>
          </div>
        ))}
      </div>

      {selectedSummary && <DatasetSummaryView summary={selectedSummary} />}
    </>
  );
}

function DatasetSummaryView({ summary }: { summary: DatasetSummary }) {
  return (
    <div className="card">
      <div className="card-header">{summary.dataset_id} — Summary</div>
      <table className="data-table">
        <tbody>
          <tr><th style={{ width: '160px' }}>Rows</th><td>{summary.row_count.toLocaleString()}</td></tr>
          <tr><th>Columns</th><td>{summary.column_count}</td></tr>
          <tr><th>Distinct Departments</th><td>{summary.distinct_departments}</td></tr>
          <tr><th>Distinct Processes</th><td>{summary.distinct_processes}</td></tr>
          <tr><th>Distinct Vendors</th><td>{summary.distinct_vendors}</td></tr>
          {summary.period_start && <tr><th>Period</th><td>{summary.period_start} to {summary.period_end}</td></tr>}
        </tbody>
      </table>

      {Object.keys(summary.numeric_measures).length > 0 && (
        <>
          <div className="card-header" style={{ marginTop: '16px' }}>Numeric Measures</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Mean</th>
                <th>Min</th>
                <th>Max</th>
                <th>Completeness</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.numeric_measures).map(([col, stats]) => (
                <tr key={col}>
                  <td>{col}</td>
                  <td>{stats.mean?.toFixed(2) ?? '—'}</td>
                  <td>{stats.min?.toFixed(2) ?? '—'}</td>
                  <td>{stats.max?.toFixed(2) ?? '—'}</td>
                  <td>{summary.completeness[col] !== undefined ? `${summary.completeness[col]}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
