"""JSON Schema adaptation per provider.

The pydantic model is the single source of truth. Providers accept different
dialects of JSON Schema:
- OpenAI strict mode (json_schema / strict tools): every object needs
  `additionalProperties: false` and ALL properties in `required`; optional
  fields are therefore expressed as nullable, and `default` is not allowed;
- Gemini `responseSchema`: an OpenAPI 3.0 subset (uppercase types, `nullable`,
  no `$ref`, no `additionalProperties`, few string formats);
- Anthropic tool `input_schema`: plain JSON Schema.

We always inline `$ref`s (pydantic emits `#/$defs/...`) because every provider
accepts inline schemas. Recursive models are rejected (not used by our tasks).

Because optional fields become nullable for OpenAI strict mode, a model may
answer `null` for a field that pydantic types as non-nullable with a default
(e.g. `flag: bool = False`); `drop_null_defaults` removes such nulls before
validation so the pydantic default applies.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

JSONSchema = dict[str, Any]


class SchemaError(ValueError):
    pass


def model_schema(model: type[BaseModel]) -> JSONSchema:
    return inline_refs(model.model_json_schema())


def inline_refs(schema: JSONSchema) -> JSONSchema:
    defs: dict[str, Any] = schema.get("$defs", {}) or {}

    def resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name in stack:
                    raise SchemaError(f"recursive schema via {name!r} is not supported")
                if name not in defs:
                    raise SchemaError(f"unresolvable $ref {ref!r}")
                target = resolve(copy.deepcopy(defs[name]), (*stack, name))
                extra = {k: resolve(v, stack) for k, v in node.items() if k != "$ref"}
                return {**target, **extra}
            return {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, stack) for v in node]
        return node

    return resolve(copy.deepcopy(schema), ())


def _allows_null(node: JSONSchema) -> bool:
    t = node.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    for key in ("anyOf", "oneOf"):
        for sub in node.get(key, []) or []:
            if isinstance(sub, dict) and _allows_null(sub):
                return True
    return False


def _make_nullable(node: JSONSchema) -> JSONSchema:
    if _allows_null(node):
        return node
    if "anyOf" in node:
        return {**node, "anyOf": [*node["anyOf"], {"type": "null"}]}
    t = node.get("type")
    if isinstance(t, str) and "enum" not in node:
        return {**node, "type": [t, "null"]}
    if isinstance(t, list):
        return {**node, "type": [*t, "null"]}
    # enums / consts / untyped: wrap
    desc = node.get("description")
    wrapped: JSONSchema = {"anyOf": [node, {"type": "null"}]}
    if desc:
        wrapped["description"] = desc
    return wrapped


# --------------------------------------------------------------------------- OpenAI strict


def openai_strict(schema: JSONSchema) -> JSONSchema:
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: JSONSchema = {}
        for k, v in node.items():
            if k in ("default", "$defs", "$schema", "properties"):
                continue
            out[k] = walk(v) if k in ("items", "anyOf", "oneOf", "allOf", "prefixItems") else v
        if node.get("type") == "object" or "properties" in node:
            props: dict[str, Any] = node.get("properties", {}) or {}
            required = set(node.get("required", []) or [])
            new_props: dict[str, Any] = {}
            for name, sub in props.items():
                adapted = walk(sub)
                new_props[name] = adapted if name in required else _make_nullable(adapted)
            out["properties"] = new_props
            out["required"] = list(new_props)
            out["additionalProperties"] = False
            out.setdefault("type", "object")
        return out

    return walk(inline_refs(schema))


# --------------------------------------------------------------------------- Gemini

_GEMINI_KEYS = {
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
_GEMINI_STRING_FORMATS = {"date-time", "enum"}


def gemini_schema(schema: JSONSchema) -> JSONSchema:
    def walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = dict(node)
        nullable = False
        # anyOf [X, null] -> X + nullable
        if "anyOf" in node:
            subs = [
                s for s in node["anyOf"] if not (isinstance(s, dict) and s.get("type") == "null")
            ]
            nullable = len(subs) != len(node["anyOf"])
            if len(subs) == 1:
                merged = {**subs[0], **{k: v for k, v in node.items() if k != "anyOf"}}
                node = merged
            else:
                node["anyOf"] = subs
        t = node.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            nullable = nullable or len(non_null) != len(t)
            node["type"] = non_null[0] if non_null else "string"
        if "const" in node:
            node["enum"] = [node.pop("const")]
            node.setdefault("type", "string")
        if "enum" in node:
            node["enum"] = [str(v) for v in node["enum"]]
            node["type"] = "string"
        fmt = node.get("format")
        if fmt is not None and (node.get("type") != "string" or fmt not in _GEMINI_STRING_FORMATS):
            node.pop("format")
            hint = f"формат: {fmt}"
            node["description"] = (
                f"{node['description']} ({hint})" if node.get("description") else hint
            )
        out: JSONSchema = {}
        for k, v in node.items():
            if k not in _GEMINI_KEYS:
                continue
            if k == "properties":
                out[k] = {name: walk(sub) for name, sub in v.items()}
                out["propertyOrdering"] = list(v)
            elif k == "items":
                out[k] = walk(v)
            elif k == "anyOf":
                out[k] = [walk(s) for s in v]
            else:
                out[k] = v
        if isinstance(out.get("type"), str):
            out["type"] = out["type"].upper()
        if nullable:
            out["nullable"] = True
        return out

    return walk(inline_refs(schema))


# --------------------------------------------------------------------------- Anthropic


def anthropic_schema(schema: JSONSchema) -> JSONSchema:
    out = inline_refs(schema)
    out.pop("$schema", None)
    out.setdefault("type", "object")
    return out


# --------------------------------------------------------------------------- validation helpers


def drop_null_defaults(data: Any, schema: JSONSchema) -> Any:
    """Remove `null` values the original schema does not allow for optional fields."""
    if isinstance(data, dict) and isinstance(schema, dict):
        props: dict[str, Any] = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        out: dict[str, Any] = {}
        for k, v in data.items():
            sub = props.get(k)
            if v is None and isinstance(sub, dict) and k not in required and not _allows_null(sub):
                continue
            out[k] = drop_null_defaults(v, sub) if isinstance(sub, dict) else v
        return out
    if (
        isinstance(data, list)
        and isinstance(schema, dict)
        and isinstance(schema.get("items"), dict)
    ):
        return [drop_null_defaults(v, schema["items"]) for v in data]
    return data
