"""Replace these hooks with a model loader and a token-streaming generator."""
import torch


def build_model(device):
    return torch.nn.Linear(4, 1).to(device)


def generate(model, messages, parameters):
    device = next(model.parameters()).device
    value = float(model(torch.ones(1, 4, device=device)).detach().cpu()[0, 0])
    for token in ("GPU ", "inference ", f"result: {value:.3f}"):
        yield token
