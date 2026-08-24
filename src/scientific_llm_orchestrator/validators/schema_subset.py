"""A small standard-library validator for the schemas used by this project.

This is intentionally a project schema-subset validator, not a full JSON
Schema implementation. It supports only the keywords used by the repository:
object/type/required/properties/additionalProperties, enum, const, pattern,
minLength, array/items/minItems, and the scalar types needed here.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence


SUPPORTED_KEYWORDS = frozenset(
    {
        "$id",
        "$schema",
        "additionalProperties",
        "const",
        "enum",
        "items",
        "minItems",
        "minLength",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
)
SCALAR_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


def _same_json_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return not (isinstance(left, float) and math.isnan(left)) and not (
            isinstance(right, float) and math.isnan(right)
        ) and left == right
    return type(left) is type(right) and left == right


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _schema_subset_errors(schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    unknown = sorted(set(schema) - SUPPORTED_KEYWORDS)
    errors.extend(f"{path}: unsupported schema keyword {key!r}" for key in unknown)

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if expected_type not in SCALAR_TYPES:
            errors.append(f"{path}: unsupported type {expected_type!r}")
    elif isinstance(expected_type, list):
        if not expected_type or any(item not in SCALAR_TYPES for item in expected_type):
            errors.append(f"{path}: type list contains an unsupported type")
    elif expected_type is not None:
        errors.append(f"{path}: type must be a string or list of strings")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            errors.append(f"{path}: properties must be an object")
        else:
            for name, child in properties.items():
                if not isinstance(name, str) or not isinstance(child, dict):
                    errors.append(f"{path}.properties: property schemas must be objects")
                else:
                    errors.extend(_schema_subset_errors(child, f"{path}.properties.{name}"))

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            errors.append(f"{path}: items must be an object schema")
        else:
            errors.extend(_schema_subset_errors(items, f"{path}.items"))

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        errors.extend(_schema_subset_errors(additional, f"{path}.additionalProperties"))
    elif additional is not None and not isinstance(additional, bool):
        errors.append(f"{path}: additionalProperties must be boolean or an object schema")
    return errors


def _validate(instance: Any, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []

    if "const" in schema and not _same_json_value(instance, schema["const"]):
        errors.append(f"{path}: value does not match const")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not any(_same_json_value(instance, item) for item in values):
            errors.append(f"{path}: value is not in enum")

    expected_type = schema.get("type")
    expected_types: Sequence[str]
    if isinstance(expected_type, str):
        expected_types = (expected_type,)
    elif isinstance(expected_type, list):
        expected_types = tuple(expected_type)
    else:
        expected_types = ()
    if expected_types and not any(_type_matches(instance, item) for item in expected_types):
        errors.append(f"{path}: expected type {expected_type!r}")
        return errors

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string is shorter than minLength")
        if "pattern" in schema:
            try:
                matched = re.search(schema["pattern"], instance)
            except (re.error, TypeError):
                errors.append(f"{path}: invalid pattern in schema")
            else:
                if matched is None:
                    errors.append(f"{path}: string does not match pattern")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: array is shorter than minItems")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(instance):
                errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in instance:
                errors.append(f"{path}: missing required property {name!r}")
        if isinstance(properties, dict):
            for name, child in properties.items():
                if name in instance and isinstance(child, dict):
                    errors.extend(_validate(instance[name], child, f"{path}.{name}"))
        additional = schema.get("additionalProperties")
        if additional is False and isinstance(properties, dict):
            for name in sorted(set(instance) - set(properties)):
                errors.append(f"{path}: additional property {name!r} is not allowed")
        elif isinstance(additional, dict) and isinstance(properties, dict):
            for name in sorted(set(instance) - set(properties)):
                errors.extend(_validate(instance[name], additional, f"{path}.{name}"))
    return errors


def validate(instance: Any, schema: Mapping[str, Any]) -> List[str]:
    """Return deterministic validation errors for the supported schema subset."""

    if not isinstance(schema, dict):
        return ["$: schema must be an object"]
    schema_errors = _schema_subset_errors(schema, "$")
    if schema_errors:
        return schema_errors
    return _validate(instance, schema, "$")


def is_valid(instance: Any, schema: Mapping[str, Any]) -> bool:
    """Return whether ``instance`` passes the project schema subset."""

    return not validate(instance, schema)
