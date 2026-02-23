import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MLP
from data_gen import make_dataset

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ddp_setup():
    dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    rank, world_size = ddp_setup()
    set_seed(args.seed + rank)

    X, y = make_dataset(n=8000, d=args.d, seed=2020)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    model = MLP(d_in=args.d)
    ddp_model = DDP(model)

    # BUG: optimizer + forward use raw model, not ddp_model => silent desync
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(args.steps):
        idx = torch.randint(0, X.shape[0], (args.batch,))
        xb = X[idx]
        yb = y[idx]

        opt.zero_grad(set_to_none=True)
        logits = model(xb)  # should be ddp_model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

    # save artifacts
    if rank == 0:
        torch.save(model.state_dict(), "artifacts_rank0.pt")
    else:
        torch.save(model.state_dict(), "artifacts_rank1.pt")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()