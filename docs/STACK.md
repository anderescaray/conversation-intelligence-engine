# The AI engineering stack in this project (and what you learn from each part)

| Layer | Tools | What you learn |
|---|---|---|
| **Speech** | faster-whisper, (pyannote) | ASR inference, diarization, real-time factor, audio formats |
| **Classical NLP** | HF Transformers, Datasets, DistilBERT, Presidio | Fine-tuning, calibration, thresholds, PII |
| **LLM application** | LangChain (chat models, structured output, tools) | Provider-agnostic calls, typed outputs, retries, cost accounting |
| **Agent orchestration** | LangGraph | State graphs, parallel branches, conditional routing, ReAct agents with tools |
| **LLMOps** | Arize Phoenix, OpenInference, MLflow (models + prompts), eval sets | Tracing, prompt versioning, offline evals, regression gates |
| **Serving** | FastAPI, Pydantic, uvicorn, Docker | Async APIs, contracts, containerisation |
| **Observability** | OpenTelemetry, Prometheus, Grafana | Metrics, traces, SLOs, dashboards, alerting |
| **Cloud and delivery** | GCP Cloud Run / AWS, Terraform, GitHub Actions | IaC, CI/CD, canary releases, rollback |
| **Engineering hygiene** | uv, ruff, pyright, pytest, pre-commit, ADRs | Professional repo standards, documented decisions |
| **AI-assisted development** | Claude Code, Devin, Pi, Cursor, git worktrees, AGENTS.md | Parallel agent workflows, task decomposition, reviewing AI code |

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
