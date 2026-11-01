import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from typing import Any, cast

import pytest

from convintel.operators.intent.descriptions import (
    AUDITED_AGAINST_REPO,
    AUDITED_AGAINST_REVISION,
    DESCRIPTIONS_VERSIONS,
    DescriptionsVersion,
    intent_descriptions,
)

# sha256 of json.dumps(list(items), ensure_ascii=False) per version, computed from the T06
# branch (scripts/synth/taxonomy.py at 6f19aea) when the descriptions moved here. Descriptions
# are immutable: a new wording is a new version with its own fingerprint, never an edit.
FINGERPRINTS: dict[DescriptionsVersion, str] = {
    1: "d59618df8a97be5f1f4fb9814aecbdd6f8e8cdfd7c322d465a4f46b729690599",
    2: "aefda66afffd494d88d4d193939a9bdbb479e8c72983749898c78eb5fa2b3c2e",
    3: "38b8ee336f21523c39b48e9f54977b78e0927bd47fee9e501806ba45efcf7baa",
}


def _fingerprint(descriptions: Mapping[str, str]) -> str:
    payload = json.dumps(list(descriptions.items()), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_versions_are_one_two_and_three() -> None:
    assert DESCRIPTIONS_VERSIONS == (1, 2, 3)


@pytest.mark.parametrize("version", DESCRIPTIONS_VERSIONS)
def test_published_versions_never_change(version: DescriptionsVersion) -> None:
    assert _fingerprint(intent_descriptions(version)) == FINGERPRINTS[version]


def test_every_version_has_the_same_77_labels_in_the_same_order() -> None:
    v1 = list(intent_descriptions(1))

    assert len(v1) == len(set(v1)) == 77
    for version in DESCRIPTIONS_VERSIONS:
        assert list(intent_descriptions(version)) == v1


def test_provenance_records_the_revision_the_descriptions_were_audited_against() -> None:
    # The T06 pin, kept as provenance. The classifier's dataset pin lives in
    # training/intent/data.py and is a different repo and revision.
    assert AUDITED_AGAINST_REPO == "PolyAI/banking77"
    assert AUDITED_AGAINST_REVISION == "796a4623935746f71378f0ebd435635a8ce08e50"


def test_labels_keep_the_exact_banking77_names() -> None:
    labels = intent_descriptions(1)

    assert "Refund_not_showing_up" in labels
    assert "reverted_card_payment?" in labels


@pytest.mark.parametrize("version", DESCRIPTIONS_VERSIONS)
def test_no_description_is_empty(version: DescriptionsVersion) -> None:
    for label, description in intent_descriptions(version).items():
        assert description.strip(), label
        assert description == description.strip(), label


def test_v3_changes_exactly_two_intents_of_v2() -> None:
    v2, v3 = intent_descriptions(2), intent_descriptions(3)

    changed = {label for label in v2 if v2[label] != v3[label]}

    assert changed == {"extra_charge_on_statement", "card_payment_not_recognised"}


def test_returned_mapping_is_read_only() -> None:
    descriptions = cast(Any, intent_descriptions(3))

    with pytest.raises(TypeError):
        descriptions["activate_my_card"] = "changed"
    with pytest.raises(TypeError):
        del descriptions["activate_my_card"]
    with pytest.raises(AttributeError):
        descriptions.clear()
    assert intent_descriptions(3)["activate_my_card"] != "changed"


def test_an_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown intent descriptions version 4"):
        intent_descriptions(cast(Any, 4))


# Runs in a fresh interpreter (this process may already have torch loaded by other tests) with an
# import hook that fails on torch/transformers, so the check holds whether or not the `ml` group
# is installed.
_IMPORT_GUARD = """
import sys

class Forbid:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("torch", "transformers"):
            raise AssertionError(f"imported {name}")
        return None

sys.meta_path.insert(0, Forbid())
import {module}
"""


def _import_guarded(module: str) -> subprocess.CompletedProcess[str]:
    code = _IMPORT_GUARD.replace("{module}", module)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_importing_descriptions_does_not_import_torch_or_transformers() -> None:
    result = _import_guarded("convintel.operators.intent.descriptions")

    assert result.returncode == 0, result.stderr


def test_the_import_guard_catches_torch() -> None:
    result = _import_guarded("torch")

    assert result.returncode != 0
    assert "AssertionError: imported torch" in result.stderr
