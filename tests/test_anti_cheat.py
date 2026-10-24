from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ENTRY = os.environ.get("SPECTRUM_DECONVOLVER", "/app/deconvolve.py")
VARIANTS = (
    {
        "catalog": [
            {"id": "anti-a", "center": 230.0, "width": 5.3, "strength": 7.4},
            {"id": "anti-b", "center": 690.0, "width": 6.7, "strength": 12.8},
        ],
        "assignment": ["probe-left", "probe-right"],
        "coefficient": 1.37,
        "baselines": {"probe-left": 2.2, "probe-right": 5.4},
        "polarity": {"probe-left": "normal", "probe-right": "normal"},
        "count": 83,
    },
    {
        "catalog": [
            {"id": "anti-c", "center": 275.0, "width": 7.1, "strength": 11.2},
            {"id": "anti-d", "center": 775.0, "width": 4.9, "strength": 6.3},
        ],
        "assignment": ["probe-right", "probe-left"],
        "coefficient": 0.83,
        "baselines": {"probe-left": 3.6, "probe-right": 1.4},
        "polarity": {"probe-left": "inverted", "probe-right": "normal"},
        "count": 97,
    },
    {
        "catalog": [
            {"id": "anti-e", "center": 190.0, "width": 5.5, "strength": 4.2},
            {"id": "anti-f", "center": 470.0, "width": 6.0, "strength": 6.8},
            {"id": "anti-g", "center": 820.0, "width": 4.7, "strength": 3.9},
        ],
        "assignment": ["probe-left", "probe-right", "probe-left"],
        "coefficient": 1.15,
        "baselines": {"probe-left": 3.0, "probe-right": 4.0},
        "polarity": {"probe-left": "inverted", "probe-right": "normal"},
        "count": 97,
    },
)


def _load_output_helpers():
    path = Path(__file__).with_name("test_outputs.py")
    spec = importlib.util.spec_from_file_location("yel018_output_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_output_helpers()


def _signature(result: dict[str, object]) -> str:
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_variants(tmp_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, variant in enumerate(VARIANTS):
        payload = HELPERS._payload(
            variant["catalog"],
            variant["assignment"],
            variant["coefficient"],
            variant["baselines"],
            count=variant["count"],
            polarity=variant["polarity"],
            run_scales=[0.7, 1.25],
        )
        source = tmp_path / f"anti-input-{index}.json"
        output = tmp_path / f"anti-output-{index}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = HELPERS._run(source, output)
        actual = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        results.append(
            {
                "returncode": completed.returncode,
                "coefficient": actual["coefficient"] if actual else None,
                "channels": HELPERS._line_channels(actual) if actual else None,
                "polarity": actual["polarity"] if actual else None,
                "signature": _signature(actual) if actual else None,
            }
        )
    return results


def test_anti_cheat_varied_spectra_cannot_reuse_one_assignment(tmp_path: Path) -> None:
    """题面：「每条目录线只落在一个探头通道中，通道归属不会写在输入里。」

    三组未公开参数改变了归属、峰高和极性；固定答案只能命中其中一组，不能同时
    满足不同的隐藏通道映射与反相探头组合。
    """

    variants = _run_variants(tmp_path)
    assert [item["returncode"] for item in variants] == [0, 0, 0]
    assert [item["coefficient"] for item in variants] == pytest.approx([1.37, 0.83, 1.15], abs=1e-6)
    assert [item["channels"] for item in variants] == [
        {"anti-a": "probe-left", "anti-b": "probe-right"},
        {"anti-c": "probe-right", "anti-d": "probe-left"},
        {"anti-e": "probe-left", "anti-f": "probe-right", "anti-g": "probe-left"},
    ]
    assert [item["polarity"] for item in variants] == [
        {"probe-left": "normal", "probe-right": "normal"},
        {"probe-left": "inverted", "probe-right": "normal"},
        {"probe-left": "inverted", "probe-right": "normal"},
    ]
    assert len({item["signature"] for item in variants}) == len(VARIANTS)
