#!/usr/bin/env python3
"""检查复现脚本是否把论文图当作数据输入。

这个脚本是 public 仓库里的一个轻量 guardrail。它只检查 src/ 和 scripts/
下会被运行的源码，确保里面没有引用本地 PDF 提取物、arXiv 图目录、图像
数字化工具，或把已有图表数据读回来的输入逻辑。换句话说，复现结果应该
来自论文方程和仿真规则，而不是来自已经画好的论文图片。
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
                errors.append(f"{rel}: 出现禁止文本 {token!r}")
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
                errors.append(f"{rel}:{node.lineno}: 出现禁止读取函数 {name}()")
    return errors


def main() -> None:
    files = tracked_files()
    errors = audit_text(files) + audit_python_calls(files)
    if errors:
        print("审计失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("审计通过：未发现源码读取论文图像或 PDF 提取物作为输入")


if __name__ == "__main__":
    main()
