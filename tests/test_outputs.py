from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ENTRY = os.environ.get("SPECTRUM_DECONVOLVER", "/app/deconvolve.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


def _run(
    payload: Path, output: Path, timeout: int = 20
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(payload), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=timeout,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _truncate_responses(payload: dict[str, object]) -> dict[str, object]:
    for run in payload["runs"]:
        for channel in run["channels"]:
            for sample in channel["samples"]:
                sample["y"] = math.floor(sample["y"] * 100.0) / 100.0
    return payload


def _trace(
    catalog: list[dict[str, object]],
    assignment: list[str],
    coefficient: float,
    baselines: dict[str, float],
    count: int = 121,
    polarity: dict[str, str] | None = None,
    scale: float = 1.0,
) -> list[dict[str, object]]:
    positions = [
        30.0 + index * 930.0 / (count - 1)
        for index in range(count)
    ]
    channels: list[dict[str, object]] = []
    for channel_id, baseline in baselines.items():
        direction = -1.0 if polarity and polarity.get(channel_id) == "inverted" else 1.0
        samples: list[dict[str, float]] = []
        for x in positions:
            y = baseline
            for line, target in zip(catalog, assignment):
                if target != channel_id:
                    continue
                center = float(line["center"])
                width = float(line["width"])
                strength = float(line["strength"])
                distance = x - center
                y += direction * coefficient * scale * strength * width * width / (
                    distance * distance + width * width
                )
            samples.append({"x": round(x, 6), "y": round(y, 9)})
        channels.append({"id": channel_id, "samples": samples})
    return channels


def _payload(
    catalog: list[dict[str, object]],
    assignment: list[str],
    coefficient: float,
    baselines: dict[str, float],
    count: int = 121,
    polarity: dict[str, str] | None = None,
    run_scales: list[float] | None = None,
) -> dict[str, object]:
    scales = run_scales or [1.0, 0.85]
    run_baselines = [baselines]
    run_baselines.extend(
        {
            channel_id: min(20.0, baseline + index)
            for channel_id, baseline in baselines.items()
        }
        for index in range(1, len(scales))
    )
    return {
        "schema": "multichannel-spectrum-v1",
        "catalog": catalog,
        "runs": [
            {
                "id": f"run-{index + 1}",
                "scale": scales[index],
                "channels": _trace(
                    catalog,
                    assignment,
                    coefficient,
                    channel_baselines,
                    count,
                    polarity,
                    scales[index],
                ),
            }
            for index, channel_baselines in enumerate(run_baselines)
        ],
    }


def _conditioned_overlap_payload() -> dict[str, object]:
    catalog = [
        {"id": "twin-b", "center": 500.0, "width": 6.4, "strength": 12.0},
        {"id": "twin-a", "center": 500.0, "width": 6.4, "strength": 12.0},
        {"id": "line-c", "center": 502.5, "width": 7.8, "strength": 7.0},
        {"id": "line-d", "center": 505.0, "width": 7.2, "strength": 15.0},
        {"id": "line-e", "center": 507.5, "width": 6.8, "strength": 6.0},
        {"id": "line-f", "center": 510.0, "width": 7.6, "strength": 9.0},
        {"id": "line-g", "center": 512.5, "width": 5.2, "strength": 4.0},
    ]
    assignment = [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "gamma",
        "alpha",
    ]
    polarity = {"alpha": 1.0, "beta": -1.0, "gamma": 1.0}
    run_specs = [
        (0.6, {"alpha": 0.1, "beta": 12.0, "gamma": 0.2}, ["gamma", "alpha", "beta"]),
        (1.1, {"alpha": 0.3, "beta": 14.0, "gamma": 0.1}, ["beta", "gamma", "alpha"]),
        (1.7, {"alpha": 0.2, "beta": 18.0, "gamma": 0.3}, ["alpha", "beta", "gamma"]),
    ]
    runs = []
    for run_index, (scale, baselines, channel_order) in enumerate(run_specs):
        channels = []
        for channel_id in channel_order:
            samples = []
            for index in range(401):
                x = 480.0 + 50.0 * index / 400.0
                signal = 0.0
                for line, target in zip(catalog, assignment):
                    if target != channel_id:
                        continue
                    distance = x - float(line["center"])
                    width = float(line["width"])
                    signal += float(line["strength"]) * width * width / (
                        distance * distance + width * width
                    )
                y = baselines[channel_id] + 0.5 * scale * polarity[channel_id] * signal
                samples.append(
                    {
                        "x": round(x, 6),
                        "y": math.floor(y * 100.0) / 100.0,
                    }
                )
            channels.append({"id": channel_id, "samples": samples})
        runs.append(
            {
                "id": f"conditioned-{run_index + 1}",
                "scale": scale,
                "channels": channels,
            }
        )
    return {
        "schema": "multichannel-spectrum-v1",
        "catalog": catalog,
        "runs": runs,
    }


def _polarity_tie_payload() -> dict[str, object]:
    catalog = [
        {
            "id": "id-00",
            "center": 15.421760784869099,
            "width": 0.5,
            "strength": 0.26,
        },
        {
            "id": "id-01",
            "center": 24.82721304600338,
            "width": 1.0,
            "strength": 1.0,
        },
    ]
    return {
        "schema": "multichannel-spectrum-v1",
        "catalog": catalog,
        "runs": [
            {
                "id": "r-0",
                "scale": 1.9999,
                "channels": [
                    {
                        "id": "ch-b",
                        "samples": [
                            {"x": 900.0 + 10.0 * index / 60.0, "y": 9.99}
                            for index in range(61)
                        ],
                    },
                    {
                        "id": "ch-a",
                        "samples": [
                            {"x": 0.1 * index / 42.0, "y": 19.89}
                            for index in range(43)
                        ],
                    },
                ],
            },
            {
                "id": "r-1",
                "scale": 1.0,
                "channels": [
                    {
                        "id": "ch-b",
                        "samples": [
                            {"x": 0.01 * index / 40.0, "y": 9.99}
                            for index in range(41)
                        ],
                    },
                    {
                        "id": "ch-a",
                        "samples": [
                            {"x": 999.9 + 0.1 * index / 40.0, "y": 9.99}
                            for index in range(41)
                        ],
                    },
                ],
            },
        ],
    }


def _read_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_channels(result: dict[str, object]) -> dict[str, str]:
    return {item["id"]: item["channel"] for item in result["peaks"]}


def test_gate_recovers_two_channel_result_contract(tmp_path: Path) -> None:
    # case_id: SCENARIO-01 ENTRY-01 RESULT-01 RESULT-02 RESULT-03 RESULT-05 RESULT-06 CATALOG-07 CATALOG-08 CATALOG-09 CATALOG-10
    """题面：
    「结果文件恰好带 `status`、`coefficient`、`baseline`、`polarity`、`peaks`、`fit_rmse`。`status` 为 `"ok"`。」
    """

    catalog = [
        {"id": "line-left", "center": 250.0, "width": 6.0, "strength": 12.0},
        {"id": "line-right", "center": 700.0, "width": 5.0, "strength": 9.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.4,
        {"probe-a": 2.5, "probe-b": 4.0},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert set(result) == {
        "baseline",
        "coefficient",
        "fit_rmse",
        "peaks",
        "polarity",
        "status",
    }
    assert result["status"] == "ok"
    assert len(result["peaks"]) == 2
    assert result["coefficient"] == pytest.approx(1.4, abs=1e-6)
    assert [peak["id"] for peak in result["peaks"]] == ["line-left", "line-right"]
    assert [peak["center"] for peak in result["peaks"]] == pytest.approx([250.0, 700.0])
    assert [peak["width"] for peak in result["peaks"]] == pytest.approx([6.0, 5.0])
    assert [peak["amplitude"] for peak in result["peaks"]] == pytest.approx([16.8, 12.6], abs=1e-5)
    assert result["polarity"] == {"probe-a": "normal", "probe-b": "normal"}
    assert result["fit_rmse"] < 1e-5


def test_hidden_channel_assignment_overrides_peak_height_order(tmp_path: Path) -> None:
    # case_id: ASSIGN-01 ASSIGN-02 COEFF-02 SOLVE-01
    """题面：
    「每条目录线只落在一个探头通道中，通道归属不会写在输入里。」
    """

    catalog = [
        {"id": "low-line", "center": 210.0, "width": 5.5, "strength": 18.0},
        {"id": "mid-line", "center": 480.0, "width": 6.5, "strength": 7.0},
        {"id": "high-line", "center": 815.0, "width": 4.5, "strength": 11.0},
    ]
    payload = _payload(
        catalog,
        ["probe-right", "probe-right", "probe-left"],
        0.85,
        {"probe-left": 3.75, "probe-right": 1.25},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    channels = _line_channels(result)
    assert completed.returncode == 0
    assert channels["low-line"] == "probe-right"
    assert channels["mid-line"] == "probe-right"
    assert channels["high-line"] == "probe-left"
    assert result["baseline"]["run-1"]["probe-left"] == pytest.approx(3.75, abs=1e-6)
    assert result["baseline"]["run-1"]["probe-right"] == pytest.approx(1.25, abs=1e-6)


def test_overlapping_lines_share_solved_response_coefficient(tmp_path: Path) -> None:
    # case_id: MODEL-01 COEFF-01 COEFF-02 COEFF-03
    """题面：
    「峰高满足 `amplitude = coefficient * strength`，极性只影响波形方向。」
    """

    catalog = [
        {"id": "line-a", "center": 430.0, "width": 7.5, "strength": 8.0},
        {"id": "line-b", "center": 512.0, "width": 5.5, "strength": 15.0},
        {"id": "line-c", "center": 795.0, "width": 4.5, "strength": 9.0},
    ]
    payload = _payload(
        catalog,
        ["probe-low", "probe-low", "probe-high"],
        2.1,
        {"probe-low": 2.0, "probe-high": 4.8},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    channels = _line_channels(result)
    peaks = {item["id"]: item for item in result["peaks"]}
    assert completed.returncode == 0
    assert result["coefficient"] == pytest.approx(2.1, abs=1e-6)
    assert channels["line-a"] == "probe-low"
    assert channels["line-b"] == "probe-low"
    assert peaks["line-a"]["amplitude"] == pytest.approx(16.8, abs=1e-5)
    assert peaks["line-b"]["amplitude"] == pytest.approx(31.5, abs=1e-5)
    assert result["fit_rmse"] < 1e-5


def test_three_channels_resolve_global_assignment(tmp_path: Path) -> None:
    # case_id: ASSIGN-03 ASSIGN-04 GLOBAL-01 RESULT-04
    """题面：
    「`peaks` 恰好有目录项数，每项恰好带 `amplitude`、`center`、`channel`、`id`、`width`，并按 `center`、`id` 依次升序排列。」
    """

    catalog = [
        {"id": "line-d", "center": 820.0, "width": 4.5, "strength": 10.0},
        {"id": "line-a", "center": 180.0, "width": 7.5, "strength": 13.0},
        {"id": "line-c", "center": 610.0, "width": 6.0, "strength": 7.5},
        {"id": "line-b", "center": 390.0, "width": 8.0, "strength": 16.0},
    ]
    payload = _payload(
        catalog,
        ["east", "north", "south", "east"],
        0.75,
        {"north": 1.0, "east": 5.5, "south": 3.0},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    channels = _line_channels(result)
    centers = [item["center"] for item in result["peaks"]]
    assert completed.returncode == 0
    assert channels["line-d"] == "east"
    assert channels["line-a"] == "north"
    assert channels["line-c"] == "south"
    assert channels["line-b"] == "east"
    assert centers == pytest.approx([180.0, 390.0, 610.0, 820.0])
    assert set(result["baseline"]) == {"run-1", "run-2"}
    assert set(result["baseline"]["run-1"]) == {"north", "east", "south"}


def test_mixed_channel_polarity_is_recovered(tmp_path: Path) -> None:
    # case_id: POLARITY-01 POLARITY-02 POLARITY-04
    """题面：
    「旧探头可能有反相接法，因此 `p_c` 取 `1` 或 `-1`。」
    """

    catalog = [
        {"id": "line-a", "center": 160.0, "width": 6.0, "strength": 6.0},
        {"id": "line-b", "center": 410.0, "width": 7.5, "strength": 6.0},
        {"id": "line-c", "center": 590.0, "width": 5.5, "strength": 10.0},
        {"id": "line-d", "center": 840.0, "width": 4.5, "strength": 7.0},
    ]
    payload = _payload(
        catalog,
        ["north", "east", "east", "south"],
        0.9,
        {"north": 1.5, "east": 12.0, "south": 3.0},
        polarity={"north": "normal", "east": "inverted", "south": "normal"},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    assert completed.returncode == 0
    assert result["polarity"] == {
        "north": "normal",
        "east": "inverted",
        "south": "normal",
    }
    assert result["coefficient"] == pytest.approx(0.9, abs=1e-6)
    assert result["fit_rmse"] < 1e-5


def test_twin_lines_use_catalog_id_canonical_tiebreak(tmp_path: Path) -> None:
    # case_id: TIE-01 TIE-02 TIE-03 TIE-04 TIE-05
    """题面：
    「若多个组合的残差在 1e-12 内并列，则按下述规范选唯一答案：将目录按 `id` 升序排列，逐项比较其通道 `id`，取字典序最小的组合；通道也相同时，`normal` 排在 `inverted` 前。」
    """

    catalog = [
        {"id": "twin-z", "center": 500.0, "width": 6.0, "strength": 10.0},
        {"id": "twin-a", "center": 500.0, "width": 6.0, "strength": 10.0},
    ]
    payload = _payload(
        catalog,
        ["zeta", "alpha"],
        1.2,
        {"zeta": 2.0, "alpha": 7.0},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    channels = _line_channels(result)
    assert completed.returncode == 0
    assert channels["twin-a"] == "alpha"
    assert channels["twin-z"] == "zeta"
    assert [peak["id"] for peak in result["peaks"]] == ["twin-a", "twin-z"]
    assert result["polarity"] == {"zeta": "normal", "alpha": "normal"}
    assert result["coefficient"] == pytest.approx(1.2, abs=1e-6)
    assert result["fit_rmse"] < 1e-5


def test_truncated_spectrum_returns_boundary_coefficient(tmp_path: Path) -> None:
    # case_id: BOUND-01 BOUND-02 BOUND-03
    """题面：
    「即使无约束解越过系数或基线边界，可行边界解仍需返回，不能直接当作无解。」
    """

    catalog = [
        {"id": "line-a", "center": 250.0, "width": 6.0, "strength": 12.0},
        {"id": "line-b", "center": 720.0, "width": 5.0, "strength": 9.0},
    ]
    payload = _truncate_responses(
        _payload(
            catalog,
            ["probe-a", "probe-b"],
            0.2,
            {"probe-a": 0.0, "probe-b": 0.0},
        )
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    assert completed.returncode == 0
    assert result["coefficient"] == pytest.approx(0.2, abs=1e-6)
    assert result["baseline"]["run-1"]["probe-a"] == pytest.approx(0.0, abs=1e-6)
    assert result["baseline"]["run-1"]["probe-b"] == pytest.approx(0.0, abs=1e-6)
    assert result["fit_rmse"] < 0.004


def test_public_upper_bound_finishes_without_nonlinear_search(tmp_path: Path) -> None:
    # case_id: SCALE-01 TIME-01 HINT-01 POLARITY-03
    """题面：
    「公开上限是一份输入含 3 个进样轮次、每轮 3 个通道和 401 个采样点，以及 7 条目录线。单次处理应在 8 秒内完成，且实现只依赖输入数据，不得读取测试目录或内置标定线。」
    """

    catalog = [
        {"id": "line-a", "center": 110.0, "width": 7.5, "strength": 14.0},
        {"id": "line-b", "center": 260.0, "width": 6.5, "strength": 8.0},
        {"id": "line-c", "center": 390.0, "width": 5.5, "strength": 17.0},
        {"id": "line-d", "center": 520.0, "width": 4.0, "strength": 11.0},
        {"id": "line-e", "center": 640.0, "width": 7.0, "strength": 9.0},
        {"id": "line-f", "center": 755.0, "width": 5.0, "strength": 13.0},
        {"id": "line-g", "center": 855.0, "width": 6.0, "strength": 7.0},
    ]
    payload = _payload(
        catalog,
        ["alpha", "beta", "gamma", "beta", "alpha", "gamma", "beta"],
        1.15,
        {"alpha": 1.5, "beta": 4.0, "gamma": 2.75},
        count=401,
        run_scales=[0.8, 1.0, 1.15],
    )
    payload["runs"][1]["channels"].reverse()
    payload["runs"][2]["channels"].reverse()
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    started = time.monotonic()
    completed = _run(source, output, timeout=20)
    elapsed = time.monotonic() - started
    result = _read_result(output)
    assert completed.returncode == 0
    assert len(result["peaks"]) == 7
    assert result["coefficient"] == pytest.approx(1.15, abs=1e-6)
    assert result["polarity"] == {
        "alpha": "normal",
        "beta": "normal",
        "gamma": "normal",
    }
    assert set(result["baseline"]) == {"run-1", "run-2", "run-3"}
    assert elapsed < 8.0


def test_channel_array_order_does_not_change_hidden_assignment(tmp_path: Path) -> None:
    # case_id: ASSIGN-05
    """题面：
    「各轮次必须使用完全相同的通道 `id` 集合。」
    """

    catalog = [
        {"id": "line-a", "center": 220.0, "width": 5.5, "strength": 8.0},
        {"id": "line-b", "center": 680.0, "width": 6.5, "strength": 11.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.25,
        {"probe-a": 2.0, "probe-b": 4.0},
        count=57,
        run_scales=[0.7, 1.4],
    )
    payload["runs"][1]["channels"].reverse()
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    channels = _line_channels(result)
    assert completed.returncode == 0
    assert channels == {"line-a": "probe-a", "line-b": "probe-b"}
    assert result["coefficient"] == pytest.approx(1.25, abs=1e-6)


def test_conditioned_overlap_recovers_canonical_joint_fit(tmp_path: Path) -> None:
    # case_id: ASSIGN-06 GLOBAL-02 MODEL-02 TIE-06
    """题面：
    「先找全部轮次合并后逐点残差平方和最小的归属与极性组合」
    """

    payload = _conditioned_overlap_payload()
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    started = time.monotonic()
    completed = _run(source, output, timeout=20)
    elapsed = time.monotonic() - started
    result = _read_result(output)
    channels = _line_channels(result)
    assert completed.returncode == 0
    assert channels == {
        "twin-a": "alpha",
        "twin-b": "beta",
        "line-c": "gamma",
        "line-d": "alpha",
        "line-e": "beta",
        "line-f": "gamma",
        "line-g": "alpha",
    }
    assert result["polarity"] == {
        "alpha": "normal",
        "beta": "inverted",
        "gamma": "normal",
    }
    assert result["coefficient"] == pytest.approx(0.499988, abs=1e-6)
    assert result["fit_rmse"] < 0.003
    assert elapsed < 8.0


def test_canonical_polarity_tie_on_constant_narrow_channel(tmp_path: Path) -> None:
    # case_id: POLARITY-05 TIE-07
    """题面：
    「若多个组合的残差在 1e-12 内并列，则按下述规范选唯一答案」
    """

    payload = _polarity_tie_payload()
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output, timeout=20)
    result = _read_result(output)
    assert completed.returncode == 0
    assert _line_channels(result) == {
        "id-00": "ch-a",
        "id-01": "ch-b",
    }
    assert result["polarity"] == {"ch-a": "normal", "ch-b": "inverted"}
    assert result["coefficient"] == pytest.approx(0.2)
    assert result["baseline"]["r-0"]["ch-a"] == pytest.approx(19.88989)
    assert result["baseline"]["r-0"]["ch-b"] == pytest.approx(9.990001)
    assert result["fit_rmse"] == pytest.approx(0.0)


def test_minimum_sample_count_is_accepted(tmp_path: Path) -> None:
    # case_id: SAMPLES-01
    """题面：
    「每个通道要有 41 至 401 个采样项，采样项恰好带 `x` 和 `y`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
        count=41,
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    assert completed.returncode == 0
    assert result["status"] == "ok"
    assert result["fit_rmse"] < 1e-5


def test_too_few_samples_are_refused(tmp_path: Path) -> None:
    # case_id: SAMPLES-02 REFUSAL-01
    """题面：
    「每个通道要有 41 至 401 个采样项，采样项恰好带 `x` 和 `y`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
        count=40,
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("REFUSED: ")
    assert completed.stderr.count("\n") == 1
    assert not output.exists()


def test_single_catalog_line_is_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-01
    """题面：
    「目录至少有两条、最多七条；`runs` 保留两轮或三轮进样，每轮接两路或三路探头。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"] = payload["catalog"][:1]
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_single_channel_is_refused(tmp_path: Path) -> None:
    # case_id: CHANNEL-02
    """题面：
    「目录至少有两条、最多七条；`runs` 保留两轮或三轮进样，每轮接两路或三路探头。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"] = payload["runs"][0]["channels"][:1]
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_run_missing_scale_is_refused(tmp_path: Path) -> None:
    # case_id: RUN-01 RUN-04 RUN-05
    """题面：
    「每个轮次对象恰好带 `id`、`scale`、`channels`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0].pop("scale")
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_run_scale_below_public_bound_is_refused(tmp_path: Path) -> None:
    # case_id: RUN-02
    """题面：
    「`scale` 的可接受区间为 0.4–2，且必须能按实数参与计算；true/false 不接受。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["scale"] = 0.399
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_run_boolean_scale_is_refused(tmp_path: Path) -> None:
    # case_id: RUN-06
    """题面：
    「`scale` 的可接受区间为 0.4–2，且必须能按实数参与计算；true/false 不接受。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["scale"] = True
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_mismatched_run_channel_sets_are_refused(tmp_path: Path) -> None:
    # case_id: RUN-03
    """题面：
    「各轮次必须使用完全相同的通道 `id` 集合。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][1]["channels"][1]["id"] = "probe-c"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_channel_identifier_is_refused(tmp_path: Path) -> None:
    # case_id: CHANNEL-01
    """题面：
    「目录项、轮次以及同一轮次内的通道 `id` 都必须是非空且各自不重复的字符串。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"][1]["id"] = "probe-a"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_catalog_identifier_is_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-04
    """题面：
    「目录项、轮次以及同一轮次内的通道 `id` 都必须是非空且各自不重复的字符串。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"][1]["id"] = "line-a"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_catalog_field_is_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-06
    """题面：
    「目录项恰好带 `id`、`center`、`width`、`strength`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"][0]["operator_note"] = "legacy"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_catalog_centers_closer_than_spacing_are_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-03
    """题面：
    「按中心排序后，相邻目录线至少相隔 2.5。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"][1]["center"] = 261.0
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_catalog_boolean_measurement_is_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-02
    """题面：
    「目录中心取 0–1000，半高半宽取 0.15–8，线强取 0.25–20，且都必须是可计算的实数；JSON 的 true/false 视为无效。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"][0]["width"] = True
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_catalog_strength_below_public_bound_is_refused(tmp_path: Path) -> None:
    # case_id: CATALOG-05
    """题面：
    「目录中心取 0–1000，半高半宽取 0.15–8，线强取 0.25–20，且都必须是可计算的实数；JSON 的 true/false 视为无效。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["catalog"][0]["strength"] = 0.249
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_catalog_lower_bound_metrics_are_accepted(tmp_path: Path) -> None:
    # case_id: CATALOG-11
    """题面：
    「目录中心取 0–1000，半高半宽取 0.15–8，线强取 0.25–20，且都必须是可计算的实数；JSON 的 true/false 视为无效。」
    """

    catalog = [
        {"id": "line-edge", "center": 0.0, "width": 0.15, "strength": 0.25},
        {"id": "line-main", "center": 500.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.0,
        {"probe-a": 2.0, "probe-b": 4.0},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = _read_result(output)
    peaks = {peak["id"]: peak for peak in result["peaks"]}
    assert completed.returncode == 0
    assert peaks["line-edge"]["center"] == pytest.approx(0.0)
    assert peaks["line-edge"]["width"] == pytest.approx(0.15)
    assert peaks["line-edge"]["amplitude"] == pytest.approx(0.25, abs=1e-5)
    assert result["status"] == "ok"


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    # case_id: SCHEMA-01
    """题面：
    「`schema` 只接受 `"multichannel-spectrum-v1"`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["schema"] = "multichannel-spectrum-v2"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_missing_top_level_key_is_refused(tmp_path: Path) -> None:
    # case_id: TOP-02
    """题面：
    「顶层只能有 `schema`、`catalog`、`runs`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload.pop("catalog")
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_missing_response_coordinate_keeps_old_result(tmp_path: Path) -> None:
    # case_id: SAMPLE-01 FAIL-01
    """题面：
    「每个通道对象恰好带 `id` 和 `samples`。每个通道要有 41 至 401 个采样项，采样项恰好带 `x` 和 `y`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"][0]["samples"][0].pop("y")
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    old_bytes = b'{"kept":true}\n'
    output.write_bytes(old_bytes)
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert output.read_bytes() == old_bytes


def test_channel_missing_samples_is_refused(tmp_path: Path) -> None:
    # case_id: CHANNEL-03
    """题面：
    「每个通道对象恰好带 `id` 和 `samples`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"][0].pop("samples")
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unsorted_sample_positions_are_refused(tmp_path: Path) -> None:
    # case_id: SAMPLE-02
    """题面：
    「同一通道内 `x` 从 0 到 1000 严格递增且不重复」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"][0]["samples"][7]["x"] = payload["runs"][0]["channels"][0]["samples"][6]["x"]
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_sample_response_above_public_bound_is_refused(tmp_path: Path) -> None:
    # case_id: SAMPLE-03
    """题面：
    「同一通道内 `x` 从 0 到 1000 严格递增且不重复；响应 `y` 不超出 -5–100。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["runs"][0]["channels"][0]["samples"][0]["y"] = 100.001
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_top_level_key_keeps_old_result(tmp_path: Path) -> None:
    # case_id: STRICT-01 TOP-01 FAIL-02
    """题面：
    「顶层只能有 `schema`、`catalog`、`runs`。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    payload["operator_note"] = "repeat"
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    old_bytes = b'{"previous":1}\n'
    output.write_bytes(old_bytes)
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert output.read_bytes() == old_bytes


def test_success_writes_silent_canonical_json(tmp_path: Path) -> None:
    # case_id: SERIAL-01 SERIAL-02 OUTPUT-01
    """题面：
    「落盘内容是紧凑 JSON，键按排序写出，末尾补一个换行。正常结束不写标准输出和标准错误；同一输入重跑必须得到相同字节。」
    """

    catalog = [
        {"id": "line-a", "center": 260.0, "width": 5.5, "strength": 10.0},
        {"id": "line-b", "center": 720.0, "width": 7.0, "strength": 12.0},
    ]
    payload = _payload(
        catalog,
        ["probe-a", "probe-b"],
        1.2,
        {"probe-a": 2.0, "probe-b": 3.0},
    )
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    second = tmp_path / "output-second.json"
    _write_json(source, payload)
    completed = _run(source, output)
    second_run = _run(source, second)
    raw = output.read_bytes()
    parsed = json.loads(raw)
    text = raw.decode("utf-8")
    key_positions = [text.index(f'"{key}"') for key in sorted(parsed)]
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert second_run.returncode == 0
    assert second.read_bytes() == raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert " " not in text
    assert key_positions == sorted(key_positions)
