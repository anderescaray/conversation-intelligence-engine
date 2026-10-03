# T04 — LLM client, prompt versioning and summary operator

**Lane:** D · **Depends on:** T00

## Goal
One well-instrumented path for every LLM call, versioned prompts, and the summary/extraction operator.

## Allowed paths
`src/convintel/llm/`, `src/convintel/operators/llm/`, `prompts/`, `tests/llm/`, `tests/operators/llm/`

## Requirements
- `convintel.llm.client`: thin async wrapper over LangChain `init_chat_model` that:
  - returns **structured output** as Pydantic models (`with_structured_output`),
  - retries with exponential backoff on rate limits/timeouts, with a global timeout,
  - counts input/output tokens and computes `cost_usd` from a price table in config,
  - emits `llm_*` metrics and relies on OpenInference auto-instrumentation for spans.
- **Providers** (via `init_chat_model`, selected only by `Settings.llm_provider` / `llm_model`):
  - **Anthropic** is the default: `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-4-5`.
  - **Ollama** for local models: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL`; cost is 0 in the price table.
  - **Google Gemini** and **OpenAI** are optional. Add their LangChain integration only if you wire
    them up, and justify the dependency in the PR.
  - A missing API key for the selected provider raises `LLMError` with a clear message, not a provider stack trace.
- `convintel.llm.prompts`: loads `prompts/<name>/v<N>.yaml` (fields: `version`, `model`,
  `temperature`, `system`, `user_template`, `notes`). Optional sync to the MLflow prompt registry.
  The prompt's `model` documents what it was tuned on; the provider and model in `Settings` always win,
  so switching provider never requires a new prompt version.
- `SummaryOperator` implementing `Operator[SummaryOutput]` using `prompts/summary/v1.yaml`.
- Prompt injection hygiene: conversation text goes inside clear delimiters, and the system prompt tells the model to treat it as data.

## Acceptance criteria
- [ ] Unit tests use a fake chat model (LangChain `FakeListChatModel` or equivalent), with no network.
- [ ] `cost_usd`, tokens, `prompt_version` and `model_id` are filled in `OperatorResult`.
- [ ] Invalid structured output triggers one repair retry, then raises `LLMError`.
- [ ] Switching provider (Anthropic ↔ Ollama, plus Gemini/OpenAI if wired up) is a config change only
  (env vars, no code edits), covered by a parametrized test over providers.

## Out of scope
Verifier node and graph (T10), eval runner (T11).
The synthetic data generator (T06) consumes this client: keep its public API small and documented.
