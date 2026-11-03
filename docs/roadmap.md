# Roadmap

## Phase 1 (Complete) ✅

- [x] Repository structure and configuration
- [x] Synthetic dataset with realistic patterns
- [x] Core configuration and logging
- [x] Domain schemas (Pydantic)
- [x] Data loader and repository
- [x] Deterministic calculation utilities
- [x] Period parsing and resolution
- [x] Thirteen analytics modules + registry
- [x] Analysis service
- [x] Evidence service
- [x] AI abstractions (interpreter, planner, explanation generator, LLM provider)
- [x] API endpoints (health, datasets, analysis, questions)
- [x] FastAPI application with middleware and error handling
- [x] Frontend shell (Dashboard, Ask, Analysis Detail, Datasets)
- [x] Unit tests for calculations, schemas, and interpreter
- [x] Documentation

## Phase 2 (Planned)

### LLM Intelligence Pipeline
- [ ] LLM-based question interpretation (replace keyword router)
- [ ] LLM-based analysis planning with reasoning traces
- [ ] Structured JSON output parsing with retry logic
- [ ] Evidence-grounded explanation generation
- [ ] Confidence scoring and uncertainty communication

### Enhanced Analytics
- [ ] Forecasting (simple moving average, exponential smoothing)
- [ ] Root-cause decomposition (multi-variate attribution)
- [ ] What-if analysis (capacity planning scenarios)
- [ ] Cross-department correlation analysis
- [ ] Vendor trend analysis over time

### Data & Infrastructure
- [ ] Incremental data refresh from source systems
- [ ] Data validation at load time with rejection reports
- [ ] Historical analysis retention
- [ ] Scheduled analysis runs

### Frontend
- [x] Dashboard trend charts (line, bar) with recharts
- [ ] Interactive charts on analysis detail pages (heatmap, drill-down)
- [ ] Period selector with comparison mode
- [ ] Evidence explorer with drill-down
- [ ] Export to PDF/CSV
- [ ] Dark mode

### Security & Reliability
- [ ] Rate limiting on question endpoint
- [ ] Request logging with redaction
- [ ] Graceful degradation when LLM is unavailable
- [ ] Caching layer for repeated questions

## Not Planned

* Kubernetes, microservices, complex auth
* Vector databases, RAG, agent frameworks
* Real ERP integrations (this is a portfolio demo)
* Enterprise data warehouses
* Multi-tenant architecture
