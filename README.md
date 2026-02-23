# RL Environments for LLM Training (AI Infrastructure)

A lightweight framework for designing and evaluating Reinforcement Learning (RL) environments used to train LLM agents on realistic AI/ML engineering tasks.

This project simulates real-world machine learning infrastructure challenges and provides automated scoring (reward signals) to evaluate whether an LLM successfully solves them.

---

## 🚀 Project Motivation

Modern LLM agents are increasingly expected to:
- Debug ML training pipelines
- Fix distributed training issues
- Understand PyTorch internals
- Operate inside real engineering constraints

This repository demonstrates how to build structured RL environments where:

1. The LLM receives a clear engineering task.
2. It edits and runs code inside a workspace.
3. An automated judge evaluates correctness.
4. A reward score is returned.

---

## 📦 Included Environment: `ddp_desync_fix`

### 🧠 Problem Description

The starter code wraps a model in **PyTorch DistributedDataParallel (DDP)** but mistakenly trains the raw model instead of the DDP-wrapped model.

This creates a **silent failure**:

- Training runs without crashing.
- Parameters drift apart across distributed ranks.
- Model weights are no longer synchronized.

This is a realistic distributed systems debugging problem.

---

## ⚙️ What the Agent Must Do

The agent must modify `train.py` so that:

- Distributed training runs with 2 processes.
- Both ranks produce synchronized model weights.
- Artifacts are saved correctly.
- No shortcuts are used (e.g., disabling DDP).

---

## 🧪 How the Judge Evaluates

The judge performs the following steps:

1. Runs:
   ```bash

   torchrun --standalone --nproc_per_node=2 train.py --steps 200

🖥 Local Setup (Windows + Anaconda)
1️⃣ Create environment
conda create -n rl_envs python=3.11 -y
conda activate rl_envs
2️⃣ Install dependencies
pip install -U pip
pip install -e .
3️⃣ Run tests
pytest -q
🌿 Branch Structure

main → Original challenge environment (buggy starter)

fix-ddp → Correct implementation that synchronizes DDP properly

🧩 Why This Environment Is Interesting

Training succeeds even when incorrect (silent failure).

Requires understanding of DistributedDataParallel internals.

Tests real ML infrastructure reasoning.

Designed to prevent trivial fixes.

CPU-friendly (runs on laptop).

🔮 Future Extensions

Continuous scoring instead of binary

Hidden evaluation datasets

FSDP-based environment

Inference optimization challenge

Runtime performance scoring

Dockerized sandbox

👨‍💻 Author

Sasi Chappidi
AI/ML Engineer – Distributed Training & ML Infrastructure
