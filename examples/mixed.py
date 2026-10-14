"""Mac layers and a remote layer participate in the same autograd graph."""
import argparse
import torch
import compute_sdk as cs

parser = argparse.ArgumentParser()
parser.add_argument("--node", required=True)
parser.add_argument("--device", default="auto")
args = parser.parse_args()
local = "mps" if torch.backends.mps.is_available() else "cpu"
with cs.connect().remote(args.node, args.device) as remote:
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(),
                               remote.linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)).to(local)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for step in range(10):
        x = torch.randn(32, 4, device=local)
        optimizer.zero_grad()
        loss = (model(x) - x.sum(1, keepdim=True)).square().mean()
        loss.backward()
        optimizer.step()
        print(step, float(loss.detach().cpu()))
