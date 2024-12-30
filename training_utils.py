import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


### Sharing alpha
def w_q(w, b, alpha):  # b is the number of bits
    w = torch.tanh(w)
    # alpha = w.data.abs().max()
    # print(alpha)
    w = torch.clamp(w / alpha, min=-1, max=1)
    w = w * (2 ** (b - 1) - 1)
    w_hat = (w.round() - w).detach() + w
    # print(torch.unique(w_hat))
    return w_hat * alpha / (2 ** (b - 1) - 1), alpha


def u_q(u, b, alpha):  # b is the number of bits
    u = torch.tanh(u)
    # alpha = w.data.abs().max()
    # print(alpha)
    u = torch.clamp(u / alpha, min=-1, max=1)
    u = u * (2 ** (b - 1) - 1)
    u_hat = (u.round() - u).detach() + u
    # print(torch.unique(w_hat))
    return u_hat * alpha / (2 ** (b - 1) - 1), u_hat


### Not sharing alpha
def b_q(w, b):  # inference
    w = torch.tanh(w)
    alpha = w.data.abs().max()
    # print(alpha)
    w = torch.clamp(w / alpha, min=-1, max=1)
    w = w * (2 ** (b - 1) - 1)
    w_hat = (w.round() - w).detach() + w
    # print(torch.unique(w_hat))
    return w_hat * alpha / (2 ** (b - 1) - 1)


def w_q_inference(w, b, alpha):
    w = torch.tanh(w)
    # alpha = w.data.abs().max()
    # print(alpha)
    w = torch.clamp(w/alpha, min=-1, max=1)
    w = w*(2**(b-1)-1)
    w_hat = w.round()
    return w_hat, alpha/(2**(b-1)-1)


def b_q_inference(w, b):
    w = torch.tanh(w)
    alpha = w.data.abs().max()
    # print(alpha)
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(b-1)-1)
    w_hat = w.round()
    return w_hat, alpha/(2**(b-1)-1)


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = sum(model(data))
            # print(type(output))
            _, idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            correct += idx.eq(target.data.view_as(idx)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def adjust_learning_rate(optimizer, cur_epoch, max_epoch):  # unused but can be used in eprop
    if (
            cur_epoch == (max_epoch * 0.5)
            or cur_epoch == (max_epoch * 0.7)
            or cur_epoch == (max_epoch * 0.9)
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 10

