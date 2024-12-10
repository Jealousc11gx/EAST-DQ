import argparse

# usage 4 mint python train_snn_ws.py --arch vgg8 -wq -uq -share -sft_rst

# python train_snn_ws.py --arch vgg8 -wq -uq -share -sft_rst && /usr/bin/shutdown
def get_args():
    parser = argparse.ArgumentParser("UQSNN")

    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--arch", default="vgg8", type=str, help="Arch vgg8")
    parser.add_argument('--dataset_dir', type=str, default='./dataset/', help='path to the dataset')
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset cifar10")
    parser.add_argument("--optim", default='adam', type=str, help="Optimizer [adam, sgd]")
    parser.add_argument('--leak_mem', default=1, type=float)
    parser.add_argument('--th', default=0.5, type=float)
    parser.add_argument('--T', type=int, default=4)  # training time steps , paper is 4 , 8 for dvs
    parser.add_argument('-uq', action='store_true')  # membrane potential quantization neuron-level
    parser.add_argument('-wq', action='store_true')  # weight quantization layer-level
    parser.add_argument('-share', action='store_true')  # training use true, testing use false
    parser.add_argument('-sft_rst', action='store_true')  # soft or hard reset
    parser.add_argument('--epoch', type=int, default=125)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
    parser.add_argument("--train_display_freq", default=1, type=int, help="display_freq for train")
    parser.add_argument("--test_display_freq", default=1, type=int, help="display_freq for test")
    parser.add_argument('--quant', default=2, type=int, help='quantization-bits')
    args = parser.parse_args()

    return args
