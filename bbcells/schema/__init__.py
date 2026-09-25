"""
JSON schemas of the input formats, and a validator for the small subset of
JSON Schema they use (type, required, properties, additionalProperties,
items, minimum). Standard library only.
"""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float),
}


def load_schema(name):
    """the schema bbcells/schema/<name>.json as a dict"""
    return json.loads((_HERE / (name + ".json")).read_text())


def validate(instance, schema, path="$"):
    """raise ValueError naming the first violation
    >>> validate({"a": [1, 2]}, {"type": "object", "properties":
    ...     {"a": {"type": "array", "items": {"type": "integer"}}}})
    >>> validate({"a": [1, "x"]}, {"type": "object", "properties":
    ...     {"a": {"type": "array", "items": {"type": "integer"}}}})
    Traceback (most recent call last):
    ...
    ValueError: $.a[1]: expected integer, got 'x'
    """
    expected = schema.get("type")
    if expected is not None:
        python_type = _TYPES[expected]
        # bool is a subclass of int, but not a JSON integer
        if not isinstance(instance, python_type) or (
                expected in ("integer", "number") and isinstance(instance, bool)):
            raise ValueError("%s: expected %s, got %r" % (path, expected, instance))
    if "minimum" in schema and instance < schema["minimum"]:
        raise ValueError("%s: %r is smaller than %r" % (path, instance, schema["minimum"]))
    if isinstance(instance, dict):
        for key in schema.get("required", ()):
            if key not in instance:
                raise ValueError("%s: missing required key %r" % (path, key))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    raise ValueError("%s: unexpected key %r" % (path, key))
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], "%s.%s" % (path, key))
    if isinstance(instance, list) and "items" in schema:
        for i, value in enumerate(instance):
            validate(value, schema["items"], "%s[%d]" % (path, i))
