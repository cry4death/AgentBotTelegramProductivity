from __future__ import annotations

from datetime import date
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from procbot.ai.schema import (
    SchemaError,
    anthropic_schema,
    drop_null_defaults,
    gemini_schema,
    inline_refs,
    model_schema,
    openai_strict,
)


class Step(BaseModel):
    title: str
    minutes: int | None = None


class Plan(BaseModel):
    kind: Literal["a", "b"]
    day: date
    steps: list[Step] = Field(min_length=1)
    note: str = "x"
    urgent: bool = False


class Node(BaseModel):
    name: str
    children: list[Node] = []


def walk(node: object):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_inline_refs_removes_defs() -> None:
    s = inline_refs(Plan.model_json_schema())
    assert "$defs" not in s and "$ref" not in str(s)
    assert s["properties"]["steps"]["items"]["properties"]["title"]["type"] == "string"


def test_recursive_schema_rejected() -> None:
    with pytest.raises(SchemaError):
        inline_refs(Node.model_json_schema())


def test_openai_strict_every_object_closed_and_all_required() -> None:
    s = openai_strict(model_schema(Plan))
    for obj in walk(s):
        if obj.get("type") == "object":
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
        assert "default" not in obj
    # originally optional fields became nullable
    assert s["properties"]["note"]["type"] == ["string", "null"]
    assert s["properties"]["urgent"]["type"] == ["boolean", "null"]
    # already-nullable stays as is, required stays non-null
    step = s["properties"]["steps"]["items"]
    assert {"type": "null"} in step["properties"]["minutes"]["anyOf"]
    assert s["properties"]["kind"] == {"enum": ["a", "b"], "title": "Kind", "type": "string"}


def test_gemini_schema_subset() -> None:
    s = gemini_schema(model_schema(Plan))
    allowed = {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "properties",
        "required",
        "items",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "propertyOrdering",
    }
    for obj in walk(s):
        if "propertyOrdering" in obj or "type" in obj:
            assert set(obj) <= allowed | set(obj.get("properties", {}))
    assert s["type"] == "OBJECT"
    assert s["propertyOrdering"] == ["kind", "day", "steps", "note", "urgent"]
    assert s["properties"]["kind"] == {"type": "STRING", "enum": ["a", "b"]}
    # unsupported string format moved into the description
    assert (
        "format" not in s["properties"]["day"] and "date" in s["properties"]["day"]["description"]
    )
    minutes = s["properties"]["steps"]["items"]["properties"]["minutes"]
    assert minutes == {"type": "INTEGER", "nullable": True}
    assert s["properties"]["steps"]["minItems"] == 1


def test_anthropic_schema_plain() -> None:
    s = anthropic_schema(Plan.model_json_schema())
    assert s["type"] == "object" and "$defs" not in s


def test_drop_null_defaults_lets_pydantic_defaults_apply() -> None:
    schema = model_schema(Plan)
    raw = {
        "kind": "a",
        "day": "2026-10-26",
        "steps": [{"title": "t", "minutes": None}],
        "note": None,
        "urgent": None,
    }
    cleaned = drop_null_defaults(raw, schema)
    assert "note" not in cleaned and "urgent" not in cleaned
    assert cleaned["steps"][0]["minutes"] is None  # nullable field keeps null
    plan = Plan.model_validate(cleaned)
    assert plan.note == "x" and plan.urgent is False
