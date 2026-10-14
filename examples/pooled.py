import torch

rank, world_size = 0, 1


def configure(node_rank, size, data_path):
    global rank, world_size
    rank, world_size = node_rank, size


def build_model():
    return torch.nn.Linear(4, 1)


def make_optimizer(parameters):
    return torch.optim.SGD(parameters, lr=.05)


def train_step(model, optimizer, batch_size, steps):
    device = next(model.parameters()).device
    # Distinct reproducible streams illustrate sharding; real modules shard a dataset here.
    generator = torch.Generator().manual_seed(100 + rank)
    for _ in range(steps):
        x = torch.randn(batch_size, 4, generator=generator).to(device)
        y = x.sum(1, keepdim=True)
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
    return {"loss": float(loss.detach().cpu())}
