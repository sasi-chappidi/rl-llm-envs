from __future__ import annotations
import os, tempfile
from dataclasses import dataclass
from .core.spec import EnvironmentSpec
from .utils.fs import copytree

@dataclass
class RunResult:
    score: float
    report: str
    workspace: str

def prepare_workspace(spec: EnvironmentSpec) -> str:
    ws = tempfile.mkdtemp(prefix=f"{spec.env_id}_")
    copytree(spec.starter_dir, ws)

    with open(spec.prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    with open(os.path.join(ws, "PROMPT.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    return ws

def judge_only(spec: EnvironmentSpec, workspace: str) -> RunResult:
    s = spec.judge.score(workspace)
    return RunResult(score=s, report=spec.judge.report(), workspace=workspace)