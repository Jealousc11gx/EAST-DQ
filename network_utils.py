import torch.nn as nn
import args_config
import torch.backends.cudnn as cudnn
from statistics import mean
import math
from training_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
cudnn.deterministic = True

args = args_config.get_args()


class QConv2dLIF(nn.Module):
    """ integerate the conv2d and LIF in the inference"""

    def __init__(self, conv_module, lif_module, num_bits_w=4, num_bits_u=4):
        super(QConv2dLIF, self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module
        self.current_time_step = 0

        self.num_bits_w = num_bits_w
        self.num_bits_u = num_bits_u

        self.total_weights = conv_module.weight.numel()

        # initial_w = conv_module.weight.data.abs().max()
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2 ** (self.num_bits_w - 1) - 1)))
        # print(initial_w)
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()
        # nn.ParameterList([nn.Parameter(initial_w) for i in range(1)]).cuda()
        # print(self.scaling[0])
        # self.scaling = nn.Parameter(,requires_grad=True).cuda()
        # print(self.scaling)
        # self.scaling.requires_grad_(requires_grad=True)


    def get_quantized_weights(self):
        if args.wq:
            if args.share:
                qweight, _ = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
        return qweight


    def forward(self, x):  # x - conv - lif
        # if self.training:
        if args.wq:  # wq quantize the weight
            if args.share:
                qweight, beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)  # b_q is not sharing the alpha
        else:
            qweight = self.conv_module.weight
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,  # bias is false
                     padding=self.conv_module.padding,
                     dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        if args.share:  # share alpha for training
            s = self.lif_module(x, args.share, beta, bias=0)
        else:  # not sharing alpha for interference
            s = self.lif_module(x, args.share, 0, bias=0)

        return s

