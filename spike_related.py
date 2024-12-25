import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_utils import *
import tracemalloc
import gc


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, leak=1, gamma=1.0, soft_reset=True, quant_u=False, num_bits_u=4):

        super(LIFSpike, self).__init__()

        # self.act = ZIF.apply
        self.quant_u = quant_u
        self.num_bits_u = num_bits_u

        self.thresh = thresh
        self.leak = leak
        self.gamma = gamma
        self.soft_reset = soft_reset

        self.membrane_potential = 0
        self.quantized_membrane_potential = 0
        self.real_membrane_potential = 0

        # print(self.thresh)

    def reset_mem(self):
        self.membrane_potential = 0

    def forward(self, s, share, beta, bias):

        H = s + self.membrane_potential
        grad = ((1.0 - torch.abs(H - self.thresh)).clamp(min=0))
        s = (((H - self.thresh) > 0).float() - H * grad).detach() + H * grad.detach()
        if self.soft_reset:
            U = (H - s * self.thresh) * self.leak
        else:
            U = H * self.leak * (1 - s)

        self.real_membrane_potential = U
        if self.quant_u:
            if share:
                self.membrane_potential, self.quantized_membrane_potential = u_q(U, self.num_bits_u, beta)
            else:
                self.membrane_potential = b_q(U, self.num_bits_u)
        else:
            self.membrane_potential = U

        return s

    def direct_forward(self, s, share, beta):

        H = s + self.membrane_potential

        # s = act(H-self.thresh)
        grad = ((1.0 - torch.abs(H - self.thresh)).clamp(min=0))
        s = (((H - self.thresh) > 0).float() - H * grad).detach() + H * grad.detach()
        if self.soft_reset:
            U = (H - s * self.thresh) * self.leak  # 为了画图修改过
        else:
            U = H * self.leak * (1 - s)

        if self.quant_u:
            if share:
                self.membrane_potential, self.quantized_membrane_potential = u_q(U, self.num_bits_u, beta)
            else:
                self.membrane_potential = b_q(U, self.num_bits_u)
        else:
            self.membrane_potential = U

        return s
