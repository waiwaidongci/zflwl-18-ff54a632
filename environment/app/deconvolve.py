#!/usr/bin/env python3
"""Starter entry point for multichannel spectrum analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2

    catalog = payload.get("catalog", []) if isinstance(payload, dict) else []
    channels = payload.get("channels", []) if isinstance(payload, dict) else []
    peaks = [
        {
            "amplitude": 0.0,
            "center": float(item.get("center", 0.0)),
            "channel": channels[0].get("id", "unknown") if channels else "unknown",
            "id": str(item.get("id", "")),
            "width": float(item.get("width", 1.0)),
        }
        for item in catalog
    ]
    result = {
        "baseline": {
            channel.get("id", "unknown"): 0.0 for channel in channels
        },
        "coefficient": 0.0,
        "fit_rmse": 0.0,
        "peaks": peaks,
        "status": "ok",
    }
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
