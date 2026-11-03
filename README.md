# Operations Intelligence

## AI-Powered Operations Analysis Platform

**Phynapse** — Applied AI Consulting & Engineering

Operations Intelligence is a deterministic operational analytics platform that helps businesses understand operational performance, identify bottlenecks, detect anomalies, and surface evidence-backed improvement opportunities. The LLM layer interprets validated evidence and produces narrative explanations without performing calculations.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (tested with 3.12)
- Node.js 18+
- Git

### Local Development (Full Stack)

#### Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

API docs available at `http://127.0.0.1:8000/docs`.

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI available at `http://localhost:5173`.

---

## 📦 Production Deployment

### Option 1: Docker (Recommended)

```bash
# Build
docker build -t operations-intelligence .

# Run with environment config
docker run -d \
  -p 8000:8000 -p 5173:5173 \
  -v $(pwd)/backend/.env:/app/.env:ro \
  -v $(pwd)/data:/app/data \
  operations-intelligence
```

### Option 2: Manual Deployment

#### Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"

# Set environment variables (see .env.example)
# Copy .env.example to .env and fill in values
cp .env.example .env

# Start the server
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm run build

# Serve the built frontend (serve, netlify, Vercel, etc.)
npx serve dist -l 5173
```

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

### Key Environment Variables

| Variable | Description | Default |
|---|---|---|
| `APP_ENV` | Environment name: `local` \| `staging` \| `production` | `local` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_CORS_ORIGINS` | Comma-separated allowed CORS origins | `http://localhost:5173,http://127.0.0.1:5173` |
| `DATA_DIR` | Path to CSV dataset directory | `data` |
| `LLM_BASE_URL` | OpenAI-compatible API endpoint | `https://api.openai.com/v1` |
| `LLM_API_KEY` | LLM API key (never commit!) | _(required for LLM features)_ |
| `LLM_MODEL` | Model identifier | `gpt-4o-mini` |
| `LLM_TIMEOUT` | Request timeout in seconds | `45` |
| `MAX_ANALYSIS_ROWS` | Hard cap on rows per analyzer | `200000` |
| `MIN_OBSERVATIONS_FOR_TREND` | Minimum observations for trend stats | `4` |
| `EVIDENCE_TOP_N` | Top contributors in evidence packages | `5` |
| `MAX_QUESTION_LENGTH` | Guard rail for question length | `600` |

**Important**: The `.env` file is gitignored. Never commit real credentials.

---

## 📊 API Endpoints

### Health Check

```
GET /api/health
```

Returns service status, dataset readiness, and LLM configuration.

### Datasets

```
GET /api/datasets       # List available datasets
GET /api/datasets/{id}  # Dataset summary with statistics
```

### Analysis

```
POST /api/analysis      # Run analysis with structured request
GET  /api/analysis      # List analysis runs
GET  /api/analysis/{id} # Retrieve specific analysis result
DELETE /api/analysis/{id} # Delete analysis run
```

### Questions

```
POST /api/questions     # Ask an operational question (natural language)
GET  /api/dashboard     # Dashboard data
```

---

## 📈 Data & Analytics

### Synthetic Datasets (5 CSV files)

| File | Description |
|---|---|
| `operations.csv` | Daily throughput, cycle time, queue and backlog by department and process |
| `orders.csv` | Order-level fulfilment detail |
| `sla_metrics.csv` | Monthly SLA rollup |
| `vendors.csv` | Vendor reference profiles |
| `departments.csv` | Department reference data |

### Regenerate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --out data --seed 20250101
```

### Available Analysis Types (12)

| Type | Purpose |
|---|---|
| `kpi_summary` | Headline operational metrics |
| `trend_analysis` | Direction and strength of a metric |
| `period_comparison` | What changed between two periods |
| `cycle_time_analysis` | Queue vs. processing decomposition |
| `throughput_analysis` | Completed operations per day |
| `backlog_analysis` | Open work levels and accumulation |
| `sla_analysis` | Compliance and breach exposure |
| `bottleneck_analysis` | Constraint ranking with explainable scores |
| `vendor_analysis` | Third-party delay contribution |
| `department_analysis` | Per-department scorecard |
| `anomaly_analysis` | Unusual operational patterns |
| `process_analysis` | Process-level cycle time |

---

## 🧪 Testing

```bash
cd backend
pytest -q
```

Run with coverage:

```bash
cd backend
pytest -q --cov=app
```

---

## 🔒 Security

- Secrets **only** via environment variables (`.env`, never committed)
- LLM API key **never** exposed to the frontend
- User-supplied identifiers resolved through fixed registry — no filesystem path injection
- All CSV paths verified to stay inside the configured data directory
- Structured error handling: stack traces never returned to clients
- LLM output treated as untrusted and re-validated before use

---

## 📁 Project Structure

```
operations-intelligence/
├── backend/              # FastAPI Python backend
│   ├── app/              # Application source
│   │   ├── core/         # Configuration, logging, errors
│   │   ├── data/         # CSV loader, repository
│   │   ├── analytics/    # 13+ analyzer modules
│   │   ├── ai/           # LLM abstractions
│   │   ├── services/     # Analysis, evidence services
│   │   ├── schemas/      # Pydantic models
│   │   └── main.py       # App factory, middleware
│   ├── data/             # CSV datasets
│   ├── scripts/          # Utility scripts
│   └── tests/            # Unit & integration tests
├── frontend/             # React + Vite frontend
│   ├── src/              # TypeScript source
│   ├── dist/             # Production build
│   └── package.json
├── docs/                 # Documentation
├── data/                 # Synthetic CSV datasets
├── .env.example          # Environment configuration template
├── .gitignore            # Git ignore rules
├── LICENSE
└── README.md             # You are here
```

---

## 🛠️ Development Workflow

### Code Quality

```bash
cd backend
ruff check .
ruff format .
mypy .

cd frontend
npm run lint  # if configured
npm run type-check
```

### Environment Setup (One-time)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"

# Frontend
cd frontend
npm install
```

---

## 📦 Deployment Checklist

- [ ] Copy `.env.example` to `.env` and fill in real values
- [ ] Set `APP_ENV=production`
- [ ] Configure `LLM_BASE_URL` and `LLM_API_KEY`
- [ ] Verify `DATA_DIR` points to correct dataset location
- [ ] Adjust `API_CORS_ORIGINS` for production domain(s)
- [ ] Set `LOG_LEVEL` appropriately (e.g., `WARNING` or `ERROR`)
- [ ] Run `pytest -q` to verify no regressions
- [ ] Build frontend: `cd frontend && npm run build`
- [ ] Deploy backend (Docker, Python any, or uvicorn directly)
- [ ] Deploy frontend (static hosting, CDN, or serve from backend)
- [ ] Verify `/api/health` returns `status: ok`
- [ ] Confirm LLM connectivity (if LLM features used)