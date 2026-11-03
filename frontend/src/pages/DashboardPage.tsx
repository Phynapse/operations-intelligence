import { useState, useEffect } from 'react';
import { fetchDashboard } from '../services/api';
import type { DashboardData, DashboardKPI, TrendPoint, BottleneckEntry, DepartmentOverview } from '../types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';

const CHART_COLORS = {
  primary: '#2563eb',
  secondary: '#0f172a',
  warning: '#ca8a04',
  danger: '#dc2626',
  success: '#16a34a',
  muted: '#8a8a9e',
};

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const result = await fetchDashboard();
        if (!cancelled) setData(result);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) return <div className="loading">Loading operational overview…</div>;
  if (error) return <div className="loading" style={{ color: 'var(--color-severity-critical)' }}>Failed to load dashboard: {error}</div>;
  if (!data) return null;

  return (
    <>
      <div className="page-header">
        <h2>Operational Overview</h2>
        <p>{data.period_start} to {data.period_end} — all departments and processes</p>
      </div>

      {/* KPI grid */}
      <div className="kpi-grid">
        {data.kpis.map((kpi, i) => (
          <KpiCard key={i} kpi={kpi} />
        ))}
      </div>

      {/* Trend charts - 2 column grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
        <TrendChart title="Throughput / day" data={data.throughput_trend} color={CHART_COLORS.primary} unit=" ops" />
        <TrendChart title="Average Cycle Time" data={data.cycle_time_trend} color={CHART_COLORS.warning} unit=" hrs" />
        <TrendChart title="SLA Compliance" data={data.sla_trend} color={CHART_COLORS.success} unit="%" domain={[90, 100]} />
        <TrendChart title="Closing Backlog" data={data.backlog_trend} color={CHART_COLORS.danger} unit=" items" />
      </div>

      {/* Bottlenecks */}
      {data.bottlenecks.length > 0 && (
        <div className="card">
          <div className="card-header">Top Bottleneck Processes</div>
          <p className="section-note">Ranked by average cycle time. Queue share indicates how much time is spent waiting vs. being worked on.</p>
          <BottleneckChart bottlenecks={data.bottlenecks} />
        </div>
      )}

      {/* Department overview table */}
      {data.departments.length > 0 && (
        <div className="card" style={{ marginTop: '16px' }}>
          <div className="card-header">Department Overview</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Department</th>
                <th style={{ textAlign: 'right' }}>Throughput/day</th>
                <th style={{ textAlign: 'right' }}>Cycle Time</th>
                <th style={{ textAlign: 'right' }}>SLA</th>
                <th style={{ textAlign: 'right' }}>Backlog</th>
              </tr>
            </thead>
            <tbody>
              {data.departments.map((dept) => (
                <DeptRow key={dept.department} dept={dept} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card" style={{ marginTop: '16px' }}>
        <div className="card-header">About This Dashboard</div>
        <p style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>
          All figures are computed deterministically from synthetic CSV datasets using Python/Pandas.
          No LLM was involved in producing these numbers. Generated at {new Date(data.generated_at).toLocaleString()}.
        </p>
      </div>
    </>
  );
}

function KpiCard({ kpi }: { kpi: DashboardKPI }) {
  const fmt = (v: number | null, unit: string): string => {
    if (v === null) return '—';
    if (unit === 'percent') return `${v.toFixed(1)}%`;
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };
  const unitLabel = kpi.unit === 'percent' ? '' : ` ${kpi.unit}`;

  return (
    <div className="kpi-card">
      <div className="kpi-label">{kpi.name}</div>
      <div className="kpi-value">
        {fmt(kpi.value, kpi.unit)}<span className="kpi-unit">{unitLabel}</span>
      </div>
      {kpi.period && <div className="kpi-period">{kpi.period}</div>}
    </div>
  );
}

function TrendChart({
  title,
  data,
  color,
  unit = '',
  domain,
}: {
  title: string;
  data: TrendPoint[];
  color: string;
  unit?: string;
  domain?: [number, number];
}) {
  const chartData = data.map((d) => ({
    period: d.period.replace('2025-', ''),
    value: d.value,
  }));

  return (
    <div className="card" style={{ padding: '16px' }}>
      <div className="card-header" style={{ marginBottom: '8px' }}>{title}</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e7" />
          <XAxis dataKey="period" tick={{ fontSize: 11, fill: '#8a8a9e' }} />
          <YAxis domain={domain || ['auto', 'auto']} tick={{ fontSize: 11, fill: '#8a8a9e' }} width={48} />
          <Tooltip
            formatter={(value: number) => [`${value}${unit}`, title]}
            contentStyle={{ fontSize: '12px', borderRadius: '6px', border: '1px solid #e5e5e7' }}
            labelStyle={{ fontWeight: 600 }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3, fill: color }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function BottleneckChart({ bottlenecks }: { bottlenecks: BottleneckEntry[] }) {
  const barData = bottlenecks.map((b) => ({
    name: b.name.length > 20 ? b.name.slice(0, 18) + '…' : b.name,
    fullName: b.name,
    cycle_time: b.cycle_time,
    queue_share: b.queue_share,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={barData} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 120 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e7" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: '#8a8a9e' }} unit=" hrs" />
        <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: '#5a5a6e' }} width={118} />
        <Tooltip
          formatter={(value: number, name: string) => {
            if (name === 'cycle_time') return [`${value?.toFixed(2)} hrs`, 'Cycle time'];
            return [`${value?.toFixed(1)}%`, 'Queue share'];
          }}
          contentStyle={{ fontSize: '12px', borderRadius: '6px', border: '1px solid #e5e5e7' }}
        />
        <Bar dataKey="cycle_time" radius={[0, 4, 4, 0]}>
          {barData.map((_, i) => (
            <Cell key={i} fill={i === 0 ? CHART_COLORS.danger : i === 1 ? CHART_COLORS.warning : CHART_COLORS.primary} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function DeptRow({ dept }: { dept: DepartmentOverview }) {
  const slaColor = dept.sla_compliance !== null
    ? dept.sla_compliance >= 98 ? 'var(--color-success)' : dept.sla_compliance >= 95 ? 'var(--color-severity-medium)' : 'var(--color-severity-critical)'
    : 'inherit';

  return (
    <tr>
      <td style={{ fontWeight: 500 }}>{dept.department}</td>
      <td style={{ textAlign: 'right' }}>{dept.throughput?.toFixed(1) ?? '—'}</td>
      <td style={{ textAlign: 'right' }}>{dept.cycle_time?.toFixed(2) ?? '—'} hrs</td>
      <td style={{ textAlign: 'right', color: slaColor, fontWeight: 600 }}>
        {dept.sla_compliance !== null ? `${dept.sla_compliance.toFixed(1)}%` : '—'}
      </td>
      <td style={{ textAlign: 'right' }}>{dept.backlog !== null ? dept.backlog.toLocaleString() : '—'}</td>
    </tr>
  );
}
