# `convintel.llm`

The only way convintel talks to an LLM (AGENTS.md rule 4). It provides one async client with
structured output, retries, a timeout, token and cost accounting, and Prometheus metrics, plus
versioned prompt files.

## Public API

| Name | What it is |
|---|---|
| `LLMClient(settings=None, *, chat_model=None, timeout_s=60, max_retries=3, backoff_base_s=0.5, sleep=asyncio.sleep, rand=random.random)` | The client. Share one instance. |
| `await client.structured(schema, prompt, *, operator, **variables) -> LLMResponse[schema]` | Renders the prompt and returns a validated `schema` instance. |
| `LLMResponse` | `output`, `model_id`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`. |
| `load_prompt(name, version=None)` | Loads `prompts/<name>/v<N>.yaml`. `None` loads the latest version. |
| `wrap_conversation(text)` | Puts untrusted text inside `<conversation>` delimiters. |
| `convintel.llm.testing` | `FakeStructuredChatModel`, `tool_answer`, `no_sleep` for offline tests. |

```python
from pydantic import BaseModel
from convintel.llm import LLMClient, load_prompt, wrap_conversation


class Persona(BaseModel):
    name: str
    mood: str


client = LLMClient()
prompt = load_prompt("my_prompt")  # prompts/my_prompt/v<latest>.yaml
response = await client.structured(
    Persona, prompt, operator="synthetic", conversation=wrap_conversation(text)
)
response.output, response.cost_usd, prompt.id  # Persona(...), USD or None, "my_prompt/v1"
```

## Behaviour

- **Structured output** uses LangChain `with_structured_output(schema, include_raw=True)`.
  If the answer does not fit the schema, the client sends **one** repair request that quotes
  the validation error. If the second answer also fails, it raises `LLMError`. Tokens from
  both attempts are counted.
- **Retries:** rate limits, 5xx from Anthropic (including 529 overloaded) or Ollama, timeouts,
  failed connections and connections dropped mid-response are retried with jittered
  exponential backoff (`backoff_base_s * 2**n`, scaled by a random factor in [0.5, 1.5), up
  to `max_retries`). Provider SDK retries are disabled so
  attempts are not multiplied. `timeout_s` bounds the whole call, retries and repair included.
- **Errors:** retries exhausted, provider API errors, timeouts, invalid output and a missing
  API key all raise `convintel.errors.LLMError`. Other exceptions are bugs and propagate.
- **Cost:** `pricing.PRICES` holds USD per million tokens. Ollama is free. If a model is not
  in the table, `cost_usd` is `None` (unknown, not zero) and a warning is logged.
- **Telemetry:** counters `convintel_llm_tokens_total{model,direction}` and
  `convintel_llm_cost_usd_total{model,operator}`. Provider-call spans come from OpenInference's
  LangChain instrumentation, which the telemetry lane sets up. Prompt and conversation text is
  never logged. Both counters are updated on every exit path, so failed calls (including a
  failed repair) still count towards tokens and cost.

## Providers

The provider and model come only from `Settings`. Switching needs env vars only:

| Provider | Env |
|---|---|
| Anthropic (default) | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-4-5`, `ANTHROPIC_API_KEY` |
| Ollama (local) | `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1:8b`, `OLLAMA_BASE_URL` |

`google` and `openai` are accepted by `Settings` but not wired up yet. Selecting either raises
`LLMError`. To add one, add its LangChain integration, a branch in `build_chat_model`, its
prices, and a case in the parametrized provider test.

## Prompts

`prompts/<name>/v<N>.yaml` must have the fields `version` (equal to `N`), `model`,
`temperature`, `system`, `user_template` and `notes`. Files are immutable once merged:
to change a prompt, add `v<N+1>.yaml` (ADR-0004). The prompt's `model` only records what the
prompt was tuned on. `Settings` decides which model actually runs.
