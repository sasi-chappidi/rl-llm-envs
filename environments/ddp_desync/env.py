from __future__ import annotations
import os
from rlenvs.core.spec import EnvironmentSpec
from rlenvs.tools.shell import ShellTool
from rlenvs.judges.ddp_desync_judge import DDPDesyncJudge

HERE = os.path.dirname(__file__)

def make_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        env_id="ddp_desync_fix",
        prompt_path=os.path.join(HERE, "prompt.txt"),
        starter_dir=os.path.join(HERE, "starter"),
        judge=DDPDesyncJudge(),
        tools=[ShellTool()],
        metadata={"domain": "distributed_training", "difficulty": "hard"},
    )