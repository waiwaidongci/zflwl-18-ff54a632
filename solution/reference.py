#!/usr/bin/env python3
"""Reference solver for multichannel Lorentzian line demultiplexing."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {"schema", "catalog", "runs"}
CATALOG_KEYS = {"id", "center", "width", "strength"}
CHANNEL_KEYS = {"id", "samples"}
RUN_KEYS = {"id", "scale", "channels"}
SAMPLE_KEYS = {"x", "y"}
CATALOG_MIN = 2
CATALOG_MAX = 7
RUN_MIN = 2
RUN_MAX = 3
CHANNEL_MIN = 2
CHANNEL_MAX = 3
SAMPLE_MIN = 41
SAMPLE_MAX = 401
COEFFICIENT_MIN = 0.2
COEFFICIENT_MAX = 5.0
BASELINE_MIN = 0.0
BASELINE_MAX = 20.0
SCALE_MIN = 0.4
SCALE_MAX = 2.0


class Refused(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refused("duplicate JSON key")
        result[key] = value
    return result


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Refused(f"{label} must be a finite number")
    return result


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Refused(f"{label} must be a non-empty string")
    return value


def _within(value: float, lower: float, upper: float) -> bool:
    return value >= lower and value <= upper


def parse_input(
    path: Path,
) -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[str, float, list[tuple[str, list[tuple[float, float]]]]]],
]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
        )
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level key set is invalid")
    if payload.get("schema") != "multichannel-spectrum-v1":
        raise Refused("unsupported schema")

    raw_catalog = payload.get("catalog")
    if not isinstance(raw_catalog, list) or not CATALOG_MIN <= len(raw_catalog) <= CATALOG_MAX:
        raise Refused("catalog size is outside the public interval")
    catalog: list[tuple[str, float, float, float]] = []
    catalog_ids: set[str] = set()
    for item in raw_catalog:
        if not isinstance(item, dict) or set(item) != CATALOG_KEYS:
            raise Refused("catalog item key set is invalid")
        line_id = _identifier(item.get("id"), "catalog id")
        if line_id in catalog_ids:
            raise Refused("catalog ids must be unique")
        center = _number(item.get("center"), "center")
        width = _number(item.get("width"), "width")
        strength = _number(item.get("strength"), "strength")
        if not _within(center, 0.0, 1000.0):
            raise Refused("center is outside the public interval")
        if not _within(width, 0.15, 8.0):
            raise Refused("width is outside the public interval")
        if not _within(strength, 0.25, 20.0):
            raise Refused("strength is outside the public interval")
        catalog_ids.add(line_id)
        catalog.append((line_id, center, width, strength))
    center_groups: dict[float, list[tuple[float, float]]] = {}
    for _, center, width, strength in catalog:
        center_groups.setdefault(center, []).append((width, strength))
    twin_groups = [values for values in center_groups.values() if len(values) == 2]
    if any(len(values) > 2 for values in center_groups.values()) or len(twin_groups) > 1:
        raise Refused("catalog contains too many coincident lines")
    if any(values[0] != values[1] for values in twin_groups):
        raise Refused("coincident catalog lines must be identical twins")
    centers = sorted(center_groups)
    if any(right - left < 2.5 for left, right in zip(centers, centers[1:])):
        raise Refused("catalog centers are too close")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not RUN_MIN <= len(raw_runs) <= RUN_MAX:
        raise Refused("run count is outside the public interval")
    runs: list[tuple[str, float, list[tuple[str, list[tuple[float, float]]]]]] = []
    run_ids: set[str] = set()
    expected_channels: set[str] | None = None
    for run_item in raw_runs:
        if not isinstance(run_item, dict) or set(run_item) != RUN_KEYS:
            raise Refused("run key set is invalid")
        run_id = _identifier(run_item.get("id"), "run id")
        if run_id in run_ids:
            raise Refused("run ids must be unique")
        scale = _number(run_item.get("scale"), "run scale")
        if not _within(scale, SCALE_MIN, SCALE_MAX):
            raise Refused("run scale is outside the public interval")
        raw_channels = run_item.get("channels")
        if not isinstance(raw_channels, list) or not CHANNEL_MIN <= len(raw_channels) <= CHANNEL_MAX:
            raise Refused("channel size is outside the public interval")
        channels: list[tuple[str, list[tuple[float, float]]]] = []
        channel_ids: set[str] = set()
        for item in raw_channels:
            if not isinstance(item, dict) or set(item) != CHANNEL_KEYS:
                raise Refused("channel item key set is invalid")
            channel_id = _identifier(item.get("id"), "channel id")
            if channel_id in channel_ids:
                raise Refused("channel ids must be unique within a run")
            raw_samples = item.get("samples")
            if not isinstance(raw_samples, list) or not SAMPLE_MIN <= len(raw_samples) <= SAMPLE_MAX:
                raise Refused("sample count is outside the public interval")
            samples: list[tuple[float, float]] = []
            previous = -math.inf
            for sample in raw_samples:
                if not isinstance(sample, dict) or set(sample) != SAMPLE_KEYS:
                    raise Refused("sample key set is invalid")
                x = _number(sample.get("x"), "sample x")
                y = _number(sample.get("y"), "sample y")
                if not 0.0 <= x <= 1000.0 or x <= previous:
                    raise Refused("sample positions must increase within the public range")
                if not _within(y, -5.0, 100.0):
                    raise Refused("sample response is outside the public range")
                samples.append((x, y))
                previous = x
            channel_ids.add(channel_id)
            channels.append((channel_id, samples))
        if expected_channels is None:
            expected_channels = channel_ids
        elif channel_ids != expected_channels:
            raise Refused("run channel id sets must match")
        run_ids.add(run_id)
        channels.sort(key=lambda item: item[0])
        runs.append((run_id, scale, channels))
    return catalog, runs


def _lorentzian(x: float, center: float, width: float) -> float:
    distance = x - center
    return width * width / (distance * distance + width * width)


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-13:
            raise ArithmeticError("singular normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    return [augmented[index][size] for index in range(size)]


def _fit_assignment(
    catalog: list[tuple[str, float, float, float]],
    channels: list[tuple[str, list[tuple[float, float]]]],
    assignment: tuple[int, ...],
    polarity: tuple[float, ...],
) -> tuple[float, list[float], float] | None:
    channel_count = len(channels)
    profiles: list[list[tuple[float, float]]] = []
    for channel_index, (_, samples) in enumerate(channels):
        assigned_lines = [
            catalog[line_index]
            for line_index, target in enumerate(assignment)
            if target == channel_index
        ]
        profiles.append(
            [
                (
                    x,
                    polarity[channel_index]
                    * math.fsum(
                        strength * _lorentzian(x, center, width)
                        for _, center, width, strength in assigned_lines
                    ),
                )
                for x, _ in samples
            ]
        )

    size = channel_count + 1
    normal = [[0.0] * size for _ in range(size)]
    right = [0.0] * size
    observed_squared = 0.0
    for channel_index, (_, samples) in enumerate(channels):
        for sample_index, (_, observed) in enumerate(samples):
            row = [0.0] * size
            row[channel_index] = 1.0
            row[channel_count] = profiles[channel_index][sample_index][1]
            for left in range(size):
                right[left] += row[left] * observed
                for other in range(size):
                    normal[left][other] += row[left] * row[other]
            observed_squared += observed * observed
    for index in range(size):
        normal[index][index] += 1e-13

    bounds = [(BASELINE_MIN, BASELINE_MAX)] * channel_count
    bounds.append((COEFFICIENT_MIN, COEFFICIENT_MAX))
    best: tuple[float, list[float]] | None = None
    for states in itertools.product((0, 1, 2), repeat=size):
        fixed = [0.0] * size
        free: list[int] = []
        for index, state in enumerate(states):
            if state == 0:
                free.append(index)
            else:
                fixed[index] = bounds[index][0 if state == 1 else 1]

        solution = fixed[:]
        if free:
            reduced = [[normal[row][column] for column in free] for row in free]
            reduced_right = [
                right[row]
                - math.fsum(
                    normal[row][column] * fixed[column]
                    for column in range(size)
                    if states[column] != 0
                )
                for row in free
            ]
            try:
                free_values = _solve_linear(reduced, reduced_right)
            except ArithmeticError:
                continue
            if any(
                not bounds[index][0] - 1e-9
                <= value
                <= bounds[index][1] + 1e-9
                for index, value in zip(free, free_values)
            ):
                continue
            for index, value in zip(free, free_values):
                solution[index] = value

        squared = (
            observed_squared
            - 2.0 * math.fsum(solution[index] * right[index] for index in range(size))
            + math.fsum(
                solution[left] * normal[left][other] * solution[other]
                for left in range(size)
                for other in range(size)
            )
        )
        squared = max(0.0, squared)
        if best is None or squared < best[0] - 1e-12:
            best = (squared, solution)
    if best is None:
        return None
    return best[0], best[1][:channel_count], best[1][channel_count]


def _bounded_solution(
    normal: list[list[float]],
    right: list[float],
    observed_squared: float,
    bounds: list[tuple[float, float]],
) -> tuple[float, list[float]] | None:
    size = len(bounds)
    best: tuple[float, list[float]] | None = None
    for states in itertools.product((0, 1, 2), repeat=size):
        fixed = [0.0] * size
        free: list[int] = []
        for index, state in enumerate(states):
            if state == 0:
                free.append(index)
            else:
                fixed[index] = bounds[index][0 if state == 1 else 1]

        solution = fixed[:]
        if free:
            reduced = [[normal[row][column] for column in free] for row in free]
            reduced_right = [
                right[row]
                - math.fsum(
                    normal[row][column] * fixed[column]
                    for column in range(size)
                    if states[column] != 0
                )
                for row in free
            ]
            try:
                free_values = _solve_linear(reduced, reduced_right)
            except ArithmeticError:
                continue
            if any(
                not bounds[index][0] - 1e-9
                <= value
                <= bounds[index][1] + 1e-9
                for index, value in zip(free, free_values)
            ):
                continue
            for index, value in zip(free, free_values):
                solution[index] = value

        squared = (
            observed_squared
            - 2.0 * math.fsum(solution[index] * right[index] for index in range(size))
            + math.fsum(
                solution[left] * normal[left][other] * solution[other]
                for left in range(size)
                for other in range(size)
            )
        )
        squared = max(0.0, squared)
        if best is None or squared < best[0] - 1e-12:
            best = (squared, solution)
    return best


def _prepare_subset_stats(
    catalog: list[tuple[str, float, float, float]],
    channels: list[tuple[str, list[tuple[float, float]]]],
) -> list[dict[str, Any]]:
    line_count = len(catalog)
    subset_count = 1 << line_count
    prepared: list[dict[str, Any]] = []
    for _, samples in channels:
        ys = [observed for _, observed in samples]
        basis: list[list[float]] = []
        for _, center, width, strength in catalog:
            basis.append(
                [
                    strength * _lorentzian(x, center, width)
                    for x, _ in samples
                ]
            )
        dots = [
            [
                math.fsum(left * right for left, right in zip(basis[i], basis[j]))
                for j in range(line_count)
            ]
            for i in range(line_count)
        ]
        line_sums = [math.fsum(values) for values in basis]
        line_observations = [
            math.fsum(value * observed for value, observed in zip(values, ys))
            for values in basis
        ]
        subset_sums = [0.0] * subset_count
        subset_observations = [0.0] * subset_count
        subset_squares = [0.0] * subset_count
        for mask in range(1, subset_count):
            low_bit = mask & -mask
            line_index = low_bit.bit_length() - 1
            rest = mask ^ low_bit
            subset_sums[mask] = subset_sums[rest] + line_sums[line_index]
            subset_observations[mask] = (
                subset_observations[rest] + line_observations[line_index]
            )
            subset_squares[mask] = (
                subset_squares[rest]
                + dots[line_index][line_index]
                + 2.0
                * math.fsum(
                    dots[line_index][other]
                    for other in range(line_count)
                    if rest & (1 << other)
                )
            )
        prepared.append(
            {
                "count": len(samples),
                "observed_sum": math.fsum(ys),
                "observed_squared": math.fsum(value * value for value in ys),
                "subset_sums": subset_sums,
                "subset_observations": subset_observations,
                "subset_squares": subset_squares,
            }
        )
    return prepared


def _profile_minimum(
    prepared_runs: list[list[dict[str, Any]]],
    masks: list[int],
    polarity: tuple[float, ...],
    scales: list[float],
) -> tuple[float, list[list[float]], float] | None:
    channels = []
    breakpoints = {COEFFICIENT_MIN, COEFFICIENT_MAX}
    for run_index, prepared in enumerate(prepared_runs):
        for channel_index, stats in enumerate(prepared):
            mask = masks[channel_index]
            count = stats["count"]
            observed_sum = stats["observed_sum"]
            factor = scales[run_index] * polarity[channel_index]
            signal = factor * stats["subset_sums"][mask]
            signal_observation = factor * stats["subset_observations"][mask]
            channels.append(
                (
                    count,
                    observed_sum,
                    stats["observed_squared"],
                    signal,
                    signal_observation,
                    factor * factor * stats["subset_squares"][mask],
                )
            )
            if abs(signal) > 1e-15:
                for value in (
                    observed_sum / signal,
                    (observed_sum - 20.0 * count) / signal,
                ):
                    if COEFFICIENT_MIN < value < COEFFICIENT_MAX:
                        breakpoints.add(value)

    ordered = sorted(breakpoints)
    candidates = list(ordered)
    intervals = list(zip(ordered, ordered[1:]))
    for lower, upper in intervals:
        midpoint = (lower + upper) * 0.5
        alpha = 0.0
        beta = 0.0
        for count, observed_sum, observed_squared, signal, signal_observation, signal_squared in channels:
            free_baseline = (observed_sum - midpoint * signal) / count
            if free_baseline <= BASELINE_MIN:
                alpha += signal_squared
                beta -= 2.0 * signal_observation
            elif free_baseline >= BASELINE_MAX:
                alpha += signal_squared
                beta += -2.0 * signal_observation + 40.0 * signal
            else:
                alpha += signal_squared - signal * signal / count
                beta += -2.0 * (
                    signal_observation - observed_sum * signal / count
                )
        if alpha > 1e-15:
            stationary = -beta / (2.0 * alpha)
            if lower < stationary < upper:
                candidates.append(stationary)

    best: tuple[float, float, list[float]] | None = None
    for coefficient in candidates:
        baselines = []
        squared = 0.0
        for count, observed_sum, observed_squared, signal, signal_observation, signal_squared in channels:
            baseline = min(
                BASELINE_MAX,
                max(BASELINE_MIN, (observed_sum - coefficient * signal) / count),
            )
            baselines.append(baseline)
            squared += (
                observed_squared
                - 2.0 * baseline * observed_sum
                - 2.0 * coefficient * signal_observation
                + count * baseline * baseline
                + 2.0 * coefficient * baseline * signal
                + coefficient * coefficient * signal_squared
            )
        squared = max(0.0, squared)
        if best is None or squared < best[0] - 1e-12:
            best = (squared, coefficient, baselines)
        elif abs(squared - best[0]) <= 1e-12 and coefficient < best[1]:
            best = (squared, coefficient, baselines)
    if best is None:
        return None
    channel_count = len(prepared_runs[0])
    nested = [
        best[2][index * channel_count : (index + 1) * channel_count]
        for index in range(len(prepared_runs))
    ]
    return best[0], nested, best[1]


def fit_spectrum(
    catalog: list[tuple[str, float, float, float]],
    runs: list[tuple[str, float, list[tuple[str, list[tuple[float, float]]]]]],
) -> dict[str, Any]:
    channel_count = len(runs[0][2])
    candidates: list[
        tuple[
            float,
            tuple[tuple[str, int], ...],
            tuple[int, ...],
            tuple[float, ...],
            list[list[float]],
            float,
        ]
    ] = []
    prepared_runs = [
        _prepare_subset_stats(catalog, channels)
        for _, _, channels in runs
    ]
    scales = [scale for _, scale, _ in runs]
    catalog_id_order = sorted(
        range(len(catalog)),
        key=lambda index: catalog[index][0],
    )
    for assignment in itertools.product(range(channel_count), repeat=len(catalog)):
        if len(set(assignment)) != channel_count:
            continue
        masks = [0] * channel_count
        for line_index, channel_index in enumerate(assignment):
            masks[channel_index] |= 1 << line_index
        for polarity in itertools.product((-1.0, 1.0), repeat=channel_count):
            candidate = _profile_minimum(
                prepared_runs,
                masks,
                polarity,
                scales,
            )
            if candidate is None:
                continue
            squared, baselines, coefficient = candidate
            signature = tuple(
                (
                    runs[0][2][assignment[index]][0],
                    0 if polarity[assignment[index]] > 0 else 1,
                )
                for index in catalog_id_order
            )
            candidates.append(
                (
                    squared,
                    signature,
                    assignment,
                    polarity,
                    baselines,
                    coefficient,
                )
            )
    if not candidates:
        raise Refused("no assignment closes within the public parameter bounds")

    minimum = min(item[0] for item in candidates)
    tied = [item for item in candidates if item[0] <= minimum + 1e-12]
    squared, _, assignment, polarity, baselines, coefficient = min(
        tied,
        key=lambda item: item[1],
    )
    sample_total = sum(
        len(samples)
        for _, _, channels in runs
        for _, samples in channels
    )
    peaks = [
        {
            "amplitude": round(coefficient * strength, 6),
            "center": round(center, 6),
            "channel": runs[0][2][assignment[index]][0],
            "id": line_id,
            "width": round(width, 6),
        }
        for index, (line_id, center, width, strength) in enumerate(catalog)
    ]
    peaks.sort(key=lambda item: (item["center"], item["id"]))
    return {
        "baseline": {
            run_id: {
                channel_id: round(run_baselines[index], 6)
                for index, (channel_id, _) in enumerate(channels)
            }
            for (run_id, _, channels), run_baselines in zip(runs, baselines)
        },
        "coefficient": round(coefficient, 6),
        "fit_rmse": round(math.sqrt(squared / sample_total), 6),
        "polarity": {
            channel_id: "normal" if polarity[index] > 0 else "inverted"
            for index, (channel_id, _) in enumerate(runs[0][2])
        },
        "peaks": peaks,
        "status": "ok",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        catalog, runs = parse_input(source)
        result = fit_spectrum(catalog, runs)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
