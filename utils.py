import torch

pi = torch.pi

def return_cos(n):
    return lambda x: torch.cos(n*pi*x)

def return_sin(n):
    return lambda x:torch.sin(n*pi*x)
