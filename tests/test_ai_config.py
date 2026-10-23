from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from procbot.ai.config import AIConfig, load_ai_config

EXAMPLE = Path(__file__).parent.parent / "config" / "ai.example.yaml"


def test_example_config_is_valid_and_has_no_secrets_or_real_models() -> None:
    cfg = load_ai_config(EXAMPLE)
    assert cfg is not None and cfg.configured
    providers = {c.provider for c in cfg.credentials.values()}
    assert {"openai", "openai_compatible", "anthropic", "gemini"} <= providers
    for m in cfg.models.values():
        assert m.model == "<model-id>"
        assert m.price_in_per_mtok is None and m.price_out_per_mtok is None
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "sk-" not in text and "AIza" not in text
    # open question 7: no budget limits by default
    assert cfg.budget.monthly_soft_usd is None and cfg.budget.monthly_hard_usd is None
    assert cfg.features.allow_keys_via_chat is False


def test_missing_file_is_none(tmp_path: Path) -> None:
    assert load_ai_config(tmp_path / "nope.yaml") is None


def test_defaults_are_accounting_only() -> None:
    cfg = AIConfig()
    assert not cfg.configured
    assert cfg.budget.monthly_hard_usd is None
    assert cfg.escalation.validation_failed_twice == "next_tier"


@pytest.mark.parametrize(
    "data",
    [
        {"credentials": {}, "models": {"m": {"credential": "nope", "model": "x"}}},
        {
            "credentials": {"c": {"provider": "openai"}},
            "models": {"m": {"credential": "c", "model": "x"}},
            "tiers": {"fast": ["other"]},
        },
        {
            "credentials": {"c": {"provider": "openai"}},
            "models": {"m": {"credential": "c", "model": "x"}},
            "tiers": {"stt": ["m"]},  # stt model without audio_input
        },
        {"credentials": {"c": {"provider": "openai_compatible"}}},  # base_url missing
        {"credentials": {"c": {"provider": "openai", "allowed_data": ["secrets"]}}},
        {"credentials": {"c": {"provider": "nope"}}},
        {"unknown_section": {}},
    ],
)
def test_invalid_configs_rejected(data: dict) -> None:
    with pytest.raises(ValidationError):
        AIConfig.model_validate(data)


def test_allowed_data_dict_form_and_sha_stable(tmp_path: Path) -> None:
    data = {
        "credentials": {
            "c": {
                "provider": "openai",
                "api_key_env": "K",
                "allowed_data": {"message": True, "memory": False},
            }
        },
        "models": {"m": {"credential": ["c"], "model": "x", "structured_output": "native_schema"}},
        "tiers": {"fast": ["m"]},
    }
    cfg = AIConfig.model_validate(data)
    assert cfg.credentials["c"].allowed_data == ["message"]
    p = tmp_path / "ai.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg2 = load_ai_config(p)
    assert cfg2 is not None and cfg2.sha256() == cfg.sha256()
