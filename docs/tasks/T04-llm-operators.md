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
- `convintel.llm.prompts`: loads `prompts/<name>/v<N>.yaml` (fields: `version`, `model`,
  `temperature`, `system`, `user_template`, `notes`). Optional sync to the MLflow prompt registry.
- `SummaryOperator` implementing `Operator[SummaryOutput]` using `prompts/summary/v1.yaml`.
- Prompt injection hygiene: conversation text goes inside clear delimiters, and the system prompt tells the model to treat it as data.

## Acceptance criteria
- [ ] Unit tests use a fake chat model (LangChain `FakeListChatModel` or equivalent), with no network.
- [ ] `cost_usd`, tokens, `prompt_version` and `model_id` are filled in `OperatorResult`.
- [ ] Invalid structured output triggers one repair retry, then raises `LLMError`.
- [ ] Switching provider (Anthropic ↔ OpenAI) is a config change only.

## Out of scope
Verifier node and graph (T10), eval runner (T11).
