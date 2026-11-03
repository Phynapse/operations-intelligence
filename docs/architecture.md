# Architecture

## Principles

1. **LLM does not perform operational arithmetic.** Python/Pandas calculates
   every metric. The LLM only receives a curated evidence package and produces
   a narrative explanation. This reduces hallucinated numbers and makes every
   result auditable.

2. **Deterministic before probabilistic.** The analytics layer is pure,
   testable and reproducible. Given the same CSV and the same plan, it always
   produces the same output.

3. **Controlled vocabulary.** Every analysis type, metric, and dimension is an
   enum. The planner cannot invent unsupported operations.

4. **Modular monolith.** All layers live in one repository, one process.
   Boundaries are enforced by imports and interfaces, not by network calls.

5. **Evidence-first explanations.** The AI layer receives structured evidence
   (numbers with provenance) rather than raw DataFrames. It explains what the
   numbers say; it does not compute them.

## Layers

```
Frontend (React + TypeScript + Vite)
  ↓ HTTP /api/*
FastAPI (Python)
  ↓
┌─ API Routes ──────────────────────────────────┐
│  /health  /datasets  /analysis  /questions     │
└────────────────────────────────────────────────┘
  ↓
┌─ AI Abstractions ─────────────────────────────┐
│  QuestionInterpreter  →  Structured Intent    │
│  AnalysisPlanner      →  Validated Plan        │
│  ExplanationGenerator →  OperationalInsight    │
│  LLMProvider          →  chat(messages)        │
└────────────────────────────────────────────────┘
  ↓
┌─ Analysis Service ────────────────────────────┐
│  Plan construction  →  Period resolution       │
│  Context building   →  Filtered frames         │
│  Analyzer dispatch  →  Structured outcome      │
│  Evidence packaging →  Ordered, capped items   │
└────────────────────────────────────────────────┘
  ↓
┌─ Analytics Layer ─────────────────────────────┐
│  KPIAnalyzer       ThroughputAnalyzer          │
│  CycleTimeAnalyzer BacklogAnalyzer             │
│  SLAAnalyzer       BottleneckAnalyzer          │
│  VendorAnalyzer    DepartmentAnalyzer          │
│  AnomalyAnalyzer   TrendAnalyzer               │
│  ContributionAnalyzer (decomposition)          │
│  PeriodComparisonAnalyzer (composite)          │
└────────────────────────────────────────────────┘
  ↓
┌─ Data Layer ──────────────────────────────────┐
│  DatasetRepository  →  lazy, cached CSV load  │
│  FilterSpec         →  validated columns      │
│  Row limits, safe paths                        │
└────────────────────────────────────────────────┘
  ↓
CSV files (data/*.csv)
```

## Data Flow

1. **Question** → QuestionInterpreter (keyword router in Phase 1)
2. **Intent** → AnalysisPlanner → validated AnalysisPlan
3. **Plan** → AnalysisService → builds AnalysisContext with filtered frames
4. **Context** → registered Analyzer → AnalysisOutcome (evidence, KPIs, findings)
5. **Result** → EvidenceService → ordered, capped EvidencePackage
6. **Package** → ExplanationGenerator → OperationalInsight (LLM in Phase 2)
7. **Insight** → Frontend

## Key Files

```
backend/
  app/main.py                  FastAPI factory, middleware, error handlers
  app/core/config.py           Settings (env vars), LLM config
  app/core/logging.py          JSON structured logging, secret redaction
  app/schemas/operational.py   Pydantic models for CSV records
  app/schemas/analysis.py     Request, plan, evidence, result models
  app/schemas/ai.py            Intent, insight, explanation models
  app/models/enums.py          AnalysisType, Metric, Dimension, etc.
  app/data/loader.py           Safe CSV loading, dtype enforcement
  app/data/repository.py       In-memory dataset access, filtering
  app/utils/calculations.py    Deterministic numeric helpers
  app/utils/periods.py         Period parsing and resolution
  app/utils/errors.py          Exception hierarchy, status codes
  app/analytics/               Thirteen analyzers + base + registry
  app/services/                AnalysisService, EvidenceService
  app/ai/                      Interpreter, planner, generator, providers
  app/api/routes/              health, dashboard, datasets, analysis, questions
```

## Security Properties

* Secrets only through environment variables (`.env.example` in repo, `.env`
  gitignored).
* The LLM API key is never returned to the frontend.
* User-supplied dataset identifiers are resolved through a fixed registry. No
  caller can supply a filesystem path.
* All CSV paths are resolved and verified to be inside the configured data
  directory.
* No arbitrary code execution from user input — only controlled enums and
  validated strings.
* Structured error handling: stack traces are never returned to the client.
* LLM output is treated as untrusted and re-validated before use.
