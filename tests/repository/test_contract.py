from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.repository


def test_repository_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "check_repository.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_imagenet_launcher_notebook_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook_path = root / "notebooks" / "imagenet_launcher.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert ".removeprefix(" not in source
    assert "TARGET_ENV |" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            cell_source = "".join(cell["source"])
            compile(cell_source, f"<notebook cell {index}>", "exec")
            ast.parse(cell_source, filename=f"<notebook cell {index}>", feature_version=(3, 8))
            if " | " in cell_source or any(
                generic in cell_source for generic in ("dict[", "list[", "set[", "tuple[")
            ):
                assert cell_source.startswith("from __future__ import annotations\n")
