import argparse
from pathlib import Path
import torch
import compute_sdk as cs
from compute_sdk.split import SplitTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--nodes", nargs=2, required=True)
args = parser.parse_args()
cluster = cs.connect()
script = Path(__file__).with_name("stages.py")
with cluster.remote(args.nodes[0], directory=script.parent) as a, cluster.remote(args.nodes[1], directory=script.parent) as b:
    model = torch.nn.Sequential(a.module(script, 0), b.module(script, 1))
    trainer = SplitTrainer(model)
    start = trainer.restore()
    for step in range(start, start + 10):
        x = torch.randn(8, 4)
        trainer.zero_grad()
        loss = (model(x) - x.sum(1, keepdim=True)).square().mean()
        loss.backward()
        trainer.step()
        trainer.checkpoint(step + 1)
        print(step, float(loss))
