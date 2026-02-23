# RL Environments for LLM Training (AI Infrastructure)

This repo contains a small framework for defining reinforcement-learning-style environments for training LLM agents on realistic AI/ML engineering tasks.

## Included environment: DDP Desync Bug Fix (PyTorch)
The starter code wraps a model in PyTorch DDP but accidentally trains the raw model instead of the DDP-wrapped model, causing silent parameter desynchronization across ranks.

### How the evaluator works
- Runs distributed training with:
  `torchrun --standalone --nproc_per_node=2 train.py --steps 200`
- Checks artifacts from both ranks are created
- Checks weights match across ranks using L2 distance threshold

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q