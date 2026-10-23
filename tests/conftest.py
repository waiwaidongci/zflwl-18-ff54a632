from __future__ import annotations

import re
import sys
from pathlib import Path


def _case_ids(test_name: str) -> list[str]:
    source = Path(__file__).with_name("test_outputs.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    marker = f"def {test_name}("
    start = next(
        (index for index, line in enumerate(lines) if marker in line),
        None,
    )
    if start is None:
        return [test_name]
    identifiers = []
    for line in lines[start + 1 :]:
        if line.startswith("def ") or line.startswith("class "):
            break
        match = re.search(r"# case_id:\s*(.+)", line)
        if match:
            identifiers.extend(match.group(1).split())
    return identifiers or [test_name]


def pytest_runtest_logreport(report) -> None:
    if report.when != "call":
        return
    status = "PASS" if report.passed else "FAIL"
    test_name = report.nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
    for case_id in _case_ids(test_name):
        sys.__stdout__.write(f"{status} {case_id}\n")
    sys.__stdout__.flush()
