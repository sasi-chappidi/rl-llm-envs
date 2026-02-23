from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List

class Tool(Protocol):
    name: str
    def call(self, **kwargs) -> str: ...

class Judge(Protocol):
    def score(self, workspace: str) -> float: ...
    def report(self) -> str: ...

@dataclass
class EnvironmentSpec:
    env_id: str
    prompt_path: str
    starter_dir: str
    judge: Judge
    tools: List[Tool]
    public_data_dir: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None