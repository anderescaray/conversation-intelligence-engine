"""Versioned prompt files (`prompts/<name>/v<N>.yaml`), see ADR-0004.

Prompts are data, not code: they are immutable once merged, and every result records the
prompt id it was produced with. A prompt's `model` documents what it was tuned on; the provider
and model in `Settings` always win, so switching provider never requires a new prompt version.
"""

import re
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from convintel.errors import LLMError

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
"""Repo-level `prompts/` directory. Callers can pass another `root` (tests, packaged builds)."""

_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")
_OPEN_TAG = "<conversation>"
_CLOSE_TAG = "</conversation>"


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int = Field(ge=1)
    model: str
    temperature: float = Field(ge=0, le=2)
    system: str
    user_template: str
    notes: str = ""

    @property
    def id(self) -> str:
        """Stable identifier recorded as `OperatorResult.prompt_version` and in traces."""
        return f"{self.name}/v{self.version}"

    def render(self, **variables: str) -> list[BaseMessage]:
        """Build the chat messages. Variable values are inserted verbatim, never re-parsed."""
        try:
            user = self.user_template.format(**variables)
        except KeyError as exc:
            raise LLMError(f"prompt {self.id} needs variable {exc.args[0]!r}") from exc
        return [SystemMessage(content=self.system), HumanMessage(content=user)]


def wrap_conversation(text: str) -> str:
    """Delimit untrusted conversation text so the model can treat it as data.

    A transcript that contains the closing tag could otherwise end the data block early and
    smuggle instructions after it, so any delimiter inside the text is neutralised first.
    """
    safe = text.replace(_CLOSE_TAG, "</ conversation>").replace(_OPEN_TAG, "< conversation>")
    return f"{_OPEN_TAG}\n{safe}\n{_CLOSE_TAG}"


def available_versions(name: str, *, root: Path | None = None) -> list[int]:
    directory = (root or PROMPTS_DIR) / name
    if not directory.is_dir():
        return []
    matches = (_VERSION_FILE.match(p.name) for p in directory.iterdir())
    return sorted(int(m.group(1)) for m in matches if m)


def load_prompt(name: str, version: int | None = None, *, root: Path | None = None) -> Prompt:
    """Load `prompts/<name>/v<version>.yaml`; `version=None` picks the highest version."""
    if version is None:
        versions = available_versions(name, root=root)
        if not versions:
            raise LLMError(f"no prompt versions found for {name!r}")
        version = versions[-1]

    path = (root or PROMPTS_DIR) / name / f"v{version}.yaml"
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LLMError(f"prompt file not found: {name}/v{version}.yaml") from exc
    except yaml.YAMLError as exc:
        raise LLMError(f"prompt {name}/v{version}.yaml is not valid YAML") from exc

    if not isinstance(raw, dict):
        raise LLMError(f"prompt {name}/v{version}.yaml must be a mapping")
    try:
        prompt = Prompt.model_validate({**raw, "name": name})
    except ValidationError as exc:
        raise LLMError(f"prompt {name}/v{version}.yaml is invalid: {exc}") from exc

    if prompt.version != version:
        raise LLMError(
            f"prompt {name}/v{version}.yaml declares version {prompt.version}; "
            "the file name and the version field must match"
        )
    return prompt
