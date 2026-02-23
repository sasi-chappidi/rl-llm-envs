import sys
from pathlib import Path

# Add repo root to Python path so we can import env.py by path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from environments.ddp_desync.env import make_env  # now works because repo root is in sys.path
from rlenvs.runner import prepare_workspace, judge_only

def test_env_fails_initial_bug():
    spec = make_env()
    ws = prepare_workspace(spec)
    res = judge_only(spec, ws)
    assert res.score == 0.0