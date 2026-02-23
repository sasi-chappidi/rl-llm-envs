from __future__ import annotations
import os, subprocess, math
from dataclasses import dataclass
import torch

def _run(cmd: str, cwd: str, timeout_s: int = 240) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        timeout=timeout_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.stdout

def _state_distance(a: dict, b: dict) -> float:
    s = 0.0
    for k in a.keys():
        ta = a[k].float()
        tb = b[k].float()
        s += torch.sum((ta - tb) ** 2).item()
    return math.sqrt(s)

@dataclass
class DDPDesyncJudge:
    max_param_l2: float = 1e-3
    _details: str = ""

    def score(self, workspace: str) -> float:
        # Run distributed training inside workspace
        out = _run("torchrun --standalone --nproc_per_node=2 train.py --steps 200", cwd=workspace)

        if "Traceback" in out:
            self._details = "FAIL: training crashed.\n" + out[-1200:]
            return 0.0

        p0 = os.path.join(workspace, "artifacts_rank0.pt")
        p1 = os.path.join(workspace, "artifacts_rank1.pt")
        if not (os.path.exists(p0) and os.path.exists(p1)):
            self._details = "FAIL: missing artifacts_rank0.pt or artifacts_rank1.pt"
            return 0.0

        s0 = torch.load(p0, map_location="cpu")
        s1 = torch.load(p1, map_location="cpu")
        dist = _state_distance(s0, s1)

        if dist > self.max_param_l2:
            self._details = f"FAIL: parameters desynced (L2={dist:.6f})"
            return 0.0

        self._details = f"PASS: parameters synced (L2={dist:.6f})"
        return 1.0

    def report(self) -> str:
        return self._details