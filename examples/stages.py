import torch


def build_stage(index):
    torch.manual_seed(index)
    return (torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()) if index == 0
            else torch.nn.Linear(8, 1))


def make_optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=.01)
