#!/usr/bin/env python3
"""Audit that reproduction scripts do not use paper figures as data inputs.

This is a lightweight guardrail for the public reproduction repo. It checks the
tracked runnable source files under src/ and scripts/ for references to local PDF
extraction artifacts, arXiv figure directories, image digitization, or
data-loading calls that would indicate the plotted points were read from existing
figures instead of generated from the paper equations and simulations.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TEXT = [
    "arxiv_source",
    "paper_page",
    "paper_text",
    "figure/test",
    "pdftoppm",
    "pdfplumber",
    "digitize",
    "webplotdigitizer",
    "imread",
    "Image.open",
    "Li 等 - 2027",
]

FORBIDDEN_CALLS = {
    "loadtxt",
    "genfromtxt",
    "read_csv",
    "read_excel",
    "imread",
}

SOURCE_SUFFIXES = {".py"}
SOURCE_ROOTS = ("src/", "scripts/")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    files = []
    for line in result.stdout.splitlines():
        if not line.startswith(SOURCE_ROOTS):
            continue
        path = ROOT / line
        if path.suffix in SOURCE_SUFFIXES and path.is_file():
            files.append(path)
    return files


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def audit_text(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TEXT:
            if token in text and rel.as_posix() != "scripts/audit_no_figure_inputs.py":
                errors.append(f"{rel}: forbidden token {token!r}")
    return errors


def audit_python_calls(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT)
        if path.suffix != ".py" or rel.as_posix() == "scripts/audit_no_figure_inputs.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func)
            tail = name.rsplit(".", 1)[-1]
            if tail in FORBIDDEN_CALLS:
                errors.append(f"{rel}:{node.lineno}: forbidden reader call {name}()")
    return errors


def main() -> None:
    files = tracked_files()
    errors = audit_text(files) + audit_python_calls(files)
    if errors:
        print("audit failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("audit ok: no paper figure/PDF extraction inputs found in tracked source")


if __name__ == "__main__":
    main()
