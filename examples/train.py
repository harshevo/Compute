import argparse
import torch
import compute_sdk as cs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    ctx = cs.init()
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 1).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    start = ctx.restore(model, optimizer)
    x = torch.randn(64, 4, device=ctx.device)
    y = x.sum(dim=1, keepdim=True)
    for step in range(start, args.steps):
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        ctx.log(step=step, loss=float(loss.detach().cpu()), device=ctx.device)
        if (step + 1) % 5 == 0 or step + 1 == args.steps:
            ctx.checkpoint(model, optimizer, step + 1)


if __name__ == "__main__":
    main()
