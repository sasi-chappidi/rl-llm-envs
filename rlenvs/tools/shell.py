from __future__ import annotations
import subprocess
from dataclasses import dataclass

@dataclass
class ShellTool:
    name: str = "shell"

    def call(self, cwd: str, cmd: str, timeout_s: int = 120) -> str:
        blocked = ["rm -rf", "mkfs", "shutdown", "reboot", ":(){", "dd if="]
        if any(b in cmd for b in blocked):
            return "ERROR: blocked command"

        try:
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
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"