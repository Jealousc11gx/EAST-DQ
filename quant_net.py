from network_utils import *
from spike_related import LIFSpike
import torchvision.utils as vutils
args = args_config.get_args()
from spikingjelly import visualizing
import random
import torch


class Q_ShareScale_VGG8(nn.Module):
    def __init__(self, time_step, dataset):
        super(Q_ShareScale_VGG8, self).__init__()

        #### Set bitwidth for quantization
        self.num_bits_w = args.quant
        self.num_bits_u = args.quant

        #### Print out the parameters for quantization

        print("quant bw for w: " + str(self.num_bits_w))
        print("quant bw for u: " + str(self.num_bits_u))

        #### Other parameters for SNNs
        self.time_step = time_step

        #### direct spikes to be summed

        input_dim = 3

        # print(args.th)

        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1, bias=False)
        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=True,
                                   quant_u=False)

        # We employ the direct encoding technique [2] for training SNNs,
        # which has proven effective in training SNNs within a few timesteps.




        conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq,
                        num_bits_u=self.num_bits_u)
        self.ConvLif2 = QConv2dLIF(conv2, lif2, self.num_bits_w, self.num_bits_u)

        self.pool1 = nn.MaxPool2d(kernel_size=2)




        conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        lif3 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq,
                        num_bits_u=self.num_bits_u)
        self.ConvLif3 = QConv2dLIF(conv3, lif3, self.num_bits_w, self.num_bits_u)





        conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        lif4 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq,
                        num_bits_u=self.num_bits_u)
        self.ConvLif4 = QConv2dLIF(conv4, lif4, self.num_bits_w, self.num_bits_u)

        self.pool2 = nn.MaxPool2d(kernel_size=2)




        conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        lif5 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq,
                        num_bits_u=self.num_bits_u)
        self.ConvLif5 = QConv2dLIF(conv5, lif5, self.num_bits_w, self.num_bits_u)




        conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        lif6 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq,
                        num_bits_u=self.num_bits_u)
        self.ConvLif6 = QConv2dLIF(conv6, lif6, self.num_bits_w, self.num_bits_u)

        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        if dataset == 'tiny':
            size = 1
            clas = 200
        else:
            size = 1
            clas = 10
        self.flatten = nn.Flatten()
        self.fc_out1 = nn.Linear(8192, 1024, bias=True)
        self.fc_out2 = nn.Linear(1024, clas, bias=True)


        self.weight_init()

    def reset_dynamics(self):  # like spikingjelly reset
        for m in self.modules():
            if isinstance(m, QConv2dLIF):
                m.lif_module.reset_mem()
        self.direct_lif.reset_mem()
        return 0

    def weight_init(self):  # kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def track_membrane_potential(self):
        membrane_potentials = {}
        for name, module in self.named_modules():
            if isinstance(module, LIFSpike):
                membrane_potentials[name] = module.membrane_potential.detach().cpu().numpy()
        return membrane_potentials


    def visualize_first_layer(self, dataset, indices, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device("cuda")
        for idx in indices:
            sample_dir = os.path.join(save_dir, f'sample_{idx}')
            os.makedirs(sample_dir, exist_ok=True)

            # 保存并放大输入图像
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)

            img_large = F.interpolate(img, scale_factor=4, mode='bilinear', align_corners=False)
            img_large = img_large.squeeze(0).cpu()

            vutils.save_image(img_large, os.path.join(sample_dir, f'input.png'), normalize=True)

            # 生成脉冲序列
            self.reset_dynamics()
            spike_seq = []
            for t in range(self.time_step):
                s = self.direct_lif.direct_forward(self.conv1(img), False, 0)
                spike_seq.append(s.detach().cpu())

            # 随机选择5个通道
            selected_channels = random.sample(range(spike_seq[0].size(1)), 5)

            # 创建大图
            fig, axs = plt.subplots(7, self.time_step + 1, figsize=(5 * (self.time_step + 1), 35))
            fig.suptitle(f'Sample {idx} (Label {label}): Input and Spike Encoding')

            # 显示输入图像
            img_np = img_large.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # 归一化到0-1
            axs[0, 0].imshow(img_np)
            axs[0, 0].set_title('Input')
            axs[0, 0].axis('off')

            color_maps = ['hot', 'gray', 'viridis', 'plasma', 'inferno']

            for t in range(self.time_step):
                time_step_dir = os.path.join(sample_dir, f'time_step_{t}')
                os.makedirs(time_step_dir, exist_ok=True)

                spikes = spike_seq[t].squeeze()

                # 显示5个特定通道的脉冲
                for i, channel in enumerate(selected_channels):
                    spike_data = spikes[channel].numpy()
                    axs[i + 1, t + 1].imshow(spike_data, cmap='hot', interpolation='nearest')
                    axs[i + 1, t + 1].set_title(f'Channel {channel}, t={t}')
                    axs[i + 1, t + 1].axis('off')

                    # 保存单独的通道图
                    plt.figure(figsize=(10, 10))
                    plt.imshow(spike_data, cmap='hot', interpolation='nearest')
                    plt.title(f'Channel {channel}, t={t}')
                    plt.axis('off')
                    plt.savefig(os.path.join(time_step_dir, f'channel_{channel}.png'), dpi=300, bbox_inches='tight')
                    plt.close()

                # 计算平均池化结果
                avg_spikes = torch.mean(spikes, dim=0).numpy()

                # 显示不同颜色映射的平均池化结果
                for j, cmap in enumerate(color_maps):
                    axs[6, t + 1].imshow(avg_spikes, cmap=cmap, interpolation='nearest')
                    axs[6, t + 1].set_title(f'Avg Spikes t={t}')
                    axs[6, t + 1].axis('off')

                    # 保存单独的平均池化图
                    plt.figure(figsize=(10, 10))
                    plt.imshow(avg_spikes, cmap=cmap, interpolation='nearest')
                    plt.title("")
                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=34)  # 设置刻度标签的字体大小
                    cbar.set_ticks(np.linspace(np.min(avg_spikes), np.max(avg_spikes), num=4))
                    plt.axis('off')
                    plt.savefig(os.path.join(time_step_dir, f'avg_spikes_{cmap}.png'), dpi=300, bbox_inches='tight')
                    plt.close()

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'all_visualizations.png'), dpi=300, bbox_inches='tight')
            plt.close()

        self.reset_dynamics()
        print(f"Visualization of first layer completed. Results saved in {save_dir}")

    def visualize_encoding(self, dataset, indices, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for idx in indices:
            # 获取指定索引的图像
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)  # 添加批次维度并移到正确的设备

            # 保存输入图像
            vutils.save_image(img.cpu(), os.path.join(save_dir, f'input_{idx}_label_{label}.png'), normalize=True)

            # 生成脉冲序列
            self.reset_dynamics()
            spike_seq = []
            for t in range(self.time_step):
                s = self.direct_lif.direct_forward(self.conv1(img), False, 0)
                spike_seq.append(s.detach().cpu())

            # 累积所有时间步的脉冲
            accumulated_spikes = torch.sum(torch.stack(spike_seq), dim=0).squeeze()

            # 可视化脉冲序列
            fig, axs = plt.subplots(6, self.time_step + 1, figsize=(5 * (self.time_step + 1), 30))
            fig.suptitle(f'Sample {idx} (Label {label}): Input and Spike Encoding')

            # 显示输入图像
            img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # 归一化到0-1
            axs[0, 0].imshow(img_np)
            axs[0, 0].set_title('Input')
            axs[0, 0].axis('off')

            for i in range(2, 6):
                axs[i, 0].imshow(img_np)
                axs[i, 0].set_title('Input')
                axs[i, 0].axis('off')

            # 显示累积脉冲图
            axs[1, 0].imshow(torch.mean(accumulated_spikes, dim=0).cpu().numpy(), cmap='viridis')
            axs[1, 0].set_title('Accumulated Spikes')
            axs[1, 0].axis('off')

            plt.figure(figsize=(10, 10))
            plt.imshow(torch.mean(accumulated_spikes, dim=0).cpu().numpy(), cmap='viridis', interpolation='nearest')
            plt.title("")
            avg_spikes = torch.mean(accumulated_spikes, dim=0).cpu().numpy()
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=34)  # 设置刻度标签的字体大小
            cbar.set_ticks(np.linspace(np.min(avg_spikes), np.max(avg_spikes), num=4))

            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'avg_spikes_timesteps.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 显示每个时间步的脉冲
            for t in range(self.time_step):
                spikes = spike_seq[t].squeeze()

                # 方法1：平均池化
                avg_spikes = torch.mean(spikes, dim=0).cpu().numpy()
                axs[0, t + 1].imshow(avg_spikes, cmap='hot', interpolation='nearest')
                axs[0, t + 1].set_title(f'Avg Spikes t={t}')
                axs[0, t + 1].axis('off')

                # 方法2：最大池化 (带归一化)
                max_spikes, _ = torch.max(spikes, dim=0)
                normalized_max_spikes = (max_spikes - max_spikes.min()) / (max_spikes.max() - max_spikes.min() + 1e-8)
                axs[1, t + 1].imshow(normalized_max_spikes.cpu().numpy(), cmap='hot', interpolation='nearest')
                axs[1, t + 1].set_title(f'Max Spikes t={t}')
                axs[1, t + 1].axis('off')

                # 方法3：二值化脉冲图
                binary_spikes = (spikes > 0).float().mean(dim=0).cpu().numpy()
                axs[2, t + 1].imshow(binary_spikes, cmap='gray', interpolation='nearest')
                axs[2, t + 1].set_title(f'Binary Spikes t={t}')
                axs[2, t + 1].axis('off')

                # 方法4：脉冲频率图
                spike_freq = spikes.mean(dim=0).cpu().numpy()
                axs[3, t + 1].imshow(spike_freq, cmap='viridis', interpolation='nearest')
                axs[3, t + 1].set_title(f'Spike Frequency t={t}')
                axs[3, t + 1].axis('off')

                # 方法5：选择特定通道 (例如前2个通道)
                for i in range(2):
                    axs[4 + i, t + 1].imshow(spikes[i].cpu().numpy(), cmap='hot', interpolation='nearest')
                    axs[4 + i, t + 1].set_title(f'Channel {i} t={t}')
                    axs[4 + i, t + 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'encoding_{idx}.png'))
            plt.close()

            print(f"Visualization of sample {idx} completed. Results saved in {save_dir}")

        self.reset_dynamics()
        print(f"Visualization of all samples completed. Results saved in {save_dir}")

    def forward(self, inp):

        u_out = []
        direct_sum = []
        s_sum = []
        quantized_membrane_sum = []
        unquantized_membrane_sum = []
        self.reset_dynamics()
        static_input = self.conv1(inp)


        for t in range(self.time_step):
            s = self.direct_lif.direct_forward(static_input, False, 0)

            s = self.ConvLif2(s)

            s = self.pool1(s)

            s = self.ConvLif3(s)


            s = self.ConvLif4(s)

            s = self.pool2(s)

            s = self.ConvLif5(s)

            s = self.ConvLif6(s)

            s = self.pool3(s)

            s = s.view(s.shape[0], -1)
            s = self.fc_out1(s)
            s = self.fc_out2(s)

            u_out += [s]

        return u_out

