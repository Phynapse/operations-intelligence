import axios from 'axios';
import type {
  HealthStatus,
  DatasetInfo,
  DatasetSummary,
  AnalysisResult,
  QuestionResponse,
  DashboardData,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

/**
 * Error carrying the API's structured `error` envelope.
 *
 * A rejected question is not a transport failure: the backend answers 422 with a
 * `code` and a `message` explaining exactly what it could not do. The UI shows
 * that message rather than a generic "request failed".
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

export function toApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status ?? 0;
    const body = error.response?.data as
      | { error?: { code?: string; message?: string } }
      | undefined;
    return new ApiError(
      body?.error?.message ?? error.message,
      body?.error?.code ?? 'request_failed',
      status,
    );
  }
  return new ApiError(error instanceof Error ? error.message : String(error), 'unknown', 0);
}

export async function fetchHealth(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>('/health');
  return data;
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const { data } = await api.get<DatasetInfo[]>('/datasets');
  return data;
}

export async function fetchDatasetSummary(datasetId: string): Promise<DatasetSummary> {
  const { data } = await api.get<DatasetSummary>(`/datasets/${datasetId}`);
  return data;
}

export async function runAnalysis(params: {
  analysis_type?: string;
  metric?: string;
  dimensions?: string[];
  period?: string;
  comparison_period?: string;
  department?: string;
  process?: string;
  question?: string;
}): Promise<AnalysisResult> {
  const { data } = await api.post<AnalysisResult>('/analysis', {
    analysis_type: params.analysis_type || null,
    metric: params.metric || null,
    dimensions: params.dimensions || [],
    period: params.period || null,
    comparison_period: params.comparison_period || null,
    department: params.department || null,
    process: params.process || null,
    question: params.question || null,
  });
  return data;
}

/**
 * Ask one operational question and return the whole envelope.
 *
 * Rejects with {@link ApiError} carrying the backend's own explanation when the
 * question is out of scope, rather than a generic HTTP failure.
 */
export async function askQuestion(question: string): Promise<QuestionResponse> {
  try {
    const { data } = await api.post<QuestionResponse>('/questions', { question });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function fetchAnalysis(id: string): Promise<AnalysisResult> {
  const { data } = await api.get<AnalysisResult>(`/analysis/${id}`);
  return data;
}

export async function fetchDashboard(): Promise<DashboardData> {
  const { data } = await api.get<DashboardData>('/dashboard');
  return data;
}
