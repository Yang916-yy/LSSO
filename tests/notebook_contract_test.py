from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

from experiments.imagenet_wds_train import parse_args


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "imagenet1k_deit3_rrlsso_formal_training.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_formal_notebook_code_cells_compile_and_use_abi2_release() -> None:
    cells = _code_cells()
    for index, source in enumerate(cells):
        compile(source, f"{NOTEBOOK.name}:code-cell-{index}", "exec")
    combined = "\n".join(cells)
    assert "releases/download/v0.3.0" in combined
    assert "lsso_mathdx_runtime-0.3.0%2Btorch2110cu128" in combined
    assert "lsso_mathdx_runtime-0.3.0%2Btorch2110cu130" in combined
    assert "torch.ops.lsso_mathdx.backend_abi() == 2" in combined
    assert "tools/build_mathdx_backend.sh" not in combined
    for retired in (
        "alpha_init",
        "solve_parameterization",
        "basis_normalization",
        "rrlsso_gain_reference",
        "gain_reference_digest",
        "extended_diagnostics",
    ):
        assert retired not in combined


def test_formal_notebook_train_command_matches_trainer_parser() -> None:
    source = next(cell for cell in _code_cells() if "def train_command" in cell)
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_command"
    )
    namespace = {"sys": sys}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), NOTEBOOK.name, "exec"),
        namespace,
    )
    config = {
        "model": "deit3_base_patch16_rrlsso",
        "stage": "pretrain",
        "epochs": 800,
        "rank": 32,
        "cache_dir": "/tmp/imagenet-wds",
        "output": "/tmp/checkpoints",
        "batch_size": 512,
        "eval_batch_size": 512,
        "grad_accum": 4,
        "workers": 32,
        "eval_workers": 4,
        "seed": 0,
        "init_checkpoint": "",
    }
    command = namespace["train_command"](config, resume=True)
    args = parse_args(command[3:])
    assert args.model == config["model"]
    assert args.stage == "pretrain"
    assert args.rank == 32
    assert args.batch_size * args.grad_accum == args.effective_batch == 2048
    assert args.eval_workers == 4
    assert args.require_mathdx and args.resume
