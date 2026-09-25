"""
Raw fixed-point data as JSON (schema: bbcells/schema/fixed_points.json).

    {"name": "P^1", "dim": 1, "rank": 1,
     "points": [{"label": "0", "weights": [[1]]},
                {"label": "oo", "weights": [[-1]]}],
     "edges": [{"from": "0", "to": "oo", "weight": [1]}]}
"""

import json
from pathlib import Path

from bbcells.core import FixedPointData
from bbcells.schema import load_schema, validate


def from_dict(document):
    """FixedPointData from a JSON-like dict, validated against the schema
    >>> data = from_dict({"dim": 1, "rank": 1, "points": [
    ...     {"label": "0", "weights": [[1]]}, {"label": "oo", "weights": [[-1]]}]})
    >>> data.points, data.weights
    (('0', 'oo'), (((1,),), ((-1,),)))
    >>> from_dict({"dim": 1, "rank": 1, "points": [{"label": "0"}]})
    Traceback (most recent call last):
    ...
    ValueError: $.points[0]: missing required key 'weights'
    """
    validate(document, load_schema("fixed_points"))
    points = document["points"]
    edges = document.get("edges")
    annotations = None
    if any("annotation" in p for p in points):
        annotations = tuple(p.get("annotation", {}) for p in points)
    return FixedPointData(
        dim=document["dim"],
        rank=document["rank"],
        points=tuple(p["label"] for p in points),
        weights=tuple(tuple(tuple(w) for w in p["weights"]) for p in points),
        edges=None if edges is None else tuple(
            (e["from"], e["to"], tuple(e["weight"])) for e in edges),
        preferred_cocharacter=(None if document.get("preferred_cocharacter") is None
                               else tuple(document["preferred_cocharacter"])),
        name=document.get("name", ""),
        annotations=annotations)


def to_dict(data):
    """inverse of from_dict (labels are converted to strings)"""
    document = {"name": data.name, "dim": data.dim, "rank": data.rank}
    if data.preferred_cocharacter is not None:
        document["preferred_cocharacter"] = list(data.preferred_cocharacter)
    document["points"] = []
    for label, wts in zip(data.points, data.weights):
        entry = {"label": str(label), "weights": [list(w) for w in wts]}
        annotation = data.annotation(label)
        if annotation:
            entry["annotation"] = {str(k): _jsonable(v) for k, v in annotation.items()}
        document["points"].append(entry)
    if data.edges is not None:
        document["edges"] = [{"from": str(p), "to": str(q), "weight": list(chi)}
                             for p, q, chi in data.edges]
    return document


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def load(path):
    return from_dict(json.loads(Path(path).read_text()))


def dump(data, path=None, indent=1):
    text = json.dumps(to_dict(data), indent=indent)
    if path is not None:
        Path(path).write_text(text + "\n")
    return text
