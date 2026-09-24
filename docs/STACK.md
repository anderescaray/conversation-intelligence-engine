# The AI engineering stack in this project (and what you learn from each part)

| Layer | Tools | What you learn | Maps to the job posting |
|---|---|---|---|
| **Speech** | faster-whisper, (pyannote) | ASR inference, diarization, real-time factor, audio formats | "voice and messaging data" |
| **Classical NLP** | HF Transformers, Datasets, DistilBERT, Presidio | Fine-tuning, calibration, thresholds, PII | "NLP libraries such as Hugging Face" |
| **LLM application** | LangChain (chat models, structured output, tools) | Provider-agnostic calls, typed outputs, retries, cost accounting | "LLMs within software systems" |
| **Agent orchestration** | LangGraph | State graphs, parallel branches, conditional routing, ReAct agents with tools | "agentic frameworks such as LangGraph" |
| **LLMOps** | Arize Phoenix, OpenInference, MLflow (models + prompts), eval sets | Tracing, prompt versioning, offline evals, regression gates | "MLOps/LLMOps: testing, versioning, registry, monitoring" |
| **Serving** | FastAPI, Pydantic, uvicorn, Docker | Async APIs, contracts, containerisation | "deploying ML services" |
| **Observability** | OpenTelemetry, Prometheus, Grafana | Metrics, traces, SLOs, dashboards, alerting | "metrics, logging, telemetry… against SLOs" |
| **Cloud and delivery** | GCP Cloud Run / AWS, Terraform, GitHub Actions | IaC, CI/CD, canary releases, rollback | "cloud infrastructure", "progressive rollouts" |
| **Engineering hygiene** | uv, ruff, pyright, pytest, pre-commit, ADRs | Professional repo standards, documented decisions | "code review", "code quality" |
| **AI-assisted development** | Orca, Claude Code, Pi, Codex, AGENTS.md | Parallel agent workflows, task decomposition, reviewing AI code | How modern AI teams work |

## About LangChain

You will use LangChain, but the modern way: **LangGraph is the core** and LangChain supplies the building
blocks (chat models, `with_structured_output`, tools, message types). You will skip the old
`LLMChain` / legacy agents, which are largely obsolete. Knowing where to stop using a framework is itself a senior signal.

## Suggested learning order

1. Structured output + tool calling with LangChain (T04)
2. LangGraph basics: state, nodes, edges, conditional edges, parallel fan-out (T10)
3. Tracing with Phoenix: read your own traces until you can explain every span (T05/T10)
4. Evals: golden sets, LLM-as-judge and its pitfalls, regression gates (T11)
5. ReAct agent with tools over your own data (T16)
6. Production: SLOs, load testing, canary releases (T13–T15)
