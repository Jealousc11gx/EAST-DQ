import dill
import torch

from quant_net import *
from training_utils import *
import math
from torch.utils.tensorboard import SummaryWriter
import args_config
import torchvision
from torchvision import transforms
import numpy as np


def visualize_layer_outputs(model, data_loader, num_images=5):
    model.eval()
    device = next(model.parameters()).device

    # 创建保存可视化结果的目录
    os.makedirs('layer_visualizations', exist_ok=True)

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_images:
                break

            images = images.to(device)

            # 获取第一层(直接编码层)的输出
            first_layer_output = model.direct_lif.direct_forward(model.conv1(images), False, 0)

            # 获取第二层卷积层的输出
            second_layer_output = model.ConvLif2(first_layer_output)

            # 可视化原始图像
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0).numpy())
            plt.title('Original Image')
            plt.axis('off')

            # 可视化第一层输出
            plt.subplot(1, 3, 2)
            plt.imshow(first_layer_output[0].sum(dim=0).cpu().numpy(), cmap='hot')
            plt.title('First Layer Output')
            plt.axis('off')

            # 可视化第二层输出
            plt.subplot(1, 3, 3)
            plt.imshow(second_layer_output[0].sum(dim=0).cpu().numpy(), cmap='hot')
            plt.title('Second Layer Output')
            plt.axis('off')

            plt.savefig(f'layer_visualizations/image_{i}.png')
            plt.close()


def visualize_distribution(model, is_weight=True, quantization_setting=''):
    os.makedirs('dist_fig', exist_ok=True)
    for name, module in model.named_modules():
        if isinstance(module, QConv2dLIF):
            if is_weight:
                # pass
                data = module.get_quantized_weights().detach().cpu().numpy().flatten()
                title = f'Unquantized Weight Distribution - {name} - {quantization_setting}'
                filename = f'w_{name}_{quantization_setting}.pdf'
            else:
                data = module.lif_module.membrane_potential.detach().cpu().numpy().flatten()
                title = f'Membrane Potential Distribution - {name} - {quantization_setting}'
                filename = f'u_{name}_{quantization_setting}.pdf'

            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=64, density=True, color='#1E97B0')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(title)
            plt.savefig(os.path.join('dist_fig', filename), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def extract_quantized_weights(model):
    quantized_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, QConv2dLIF):
            w_hat, scale = w_q_inference(module.conv_module.weight, module.num_bits_w, module.beta[0])
            quantized_weights[name] = (w_hat, scale)
    return quantized_weights



def extract_thresholds(model):
    thresholds = {}
    for name, module in model.named_modules():
        if isinstance(module, QConv2dLIF):
            alpha = module.beta[0].item()
            adjusted_threshold = module.lif_module.thresh / alpha
            thresholds[name] = adjusted_threshold
    return thresholds


def save_deployment_info(model, quantization_setting):
    quantized_weights = extract_quantized_weights(model)
    adjusted_thresholds = extract_thresholds(model)

    deployment_info = {
        'weights': quantized_weights,
        'thresholds': adjusted_thresholds
    }
    print(deployment_info)
    torch.save(deployment_info, f"deployment_model_{quantization_setting}.pth")


def visualize_weights(model, quantization_setting=''):
    os.makedirs('weight_visualization', exist_ok=True)

    for name, module in model.named_modules():
        if isinstance(module, (QConv2dLIF, nn.Conv2d)):
            if isinstance(module, QConv2dLIF):
                original_weights = module.conv_module.weight.detach().cpu().numpy().flatten()
            else:
                original_weights = module.weight.detach().cpu().numpy().flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(original_weights, bins=128, density=True, color='blue', alpha=0.7, label='Original')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.title(f'Original Weight Distribution - {name} - {quantization_setting}')
            plt.legend()
            plt.savefig(f'weight_visualization/original_w_{name}_{quantization_setting}.png', bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()

            # 可视化量化后的权重
            if isinstance(module, QConv2dLIF):
                quantized_weights, _ = w_q_inference(module.conv_module.weight, module.num_bits_w, module.beta[0])
                beta_value = module.beta[0].item()
                print(f"Beta for {name}: {beta_value:.6f}, Quantized threshold is {(0.5/beta_value):.4f}")
                quantized_weights = quantized_weights.detach().cpu().numpy().flatten()
                plt.figure(figsize=(10, 6))
                plt.hist(quantized_weights, bins=32, density=True, color='red', alpha=0.7, label='Quantized')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                plt.title(f'Quantized Weight Distribution - {name} - {quantization_setting}')
                plt.legend()
                plt.savefig(f'weight_visualization/quantized_w_{name}_{quantization_setting}.png', bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()

    print("Weight visualization completed.")



def visualize_membrane_potential(model, data_loader, quantization_setting='', time_steps=4):
    model.eval()
    device = next(model.parameters()).device
    os.makedirs('membrane_visualization', exist_ok=True)
    print(f"\nStarting membrane potential visualization for {quantization_setting}")

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i > 0:  # 只处理第一个批次
                break

            images = images.to(device)
            model.reset_dynamics()  # 重置网络动态

            # 存储所有时间步的膜电位
            all_membrane_potentials = {}

            for t in range(time_steps):
                print(f"\nProcessing time step {t + 1}/{time_steps}")

                # 存储每一层的膜电位
                membrane_potentials = {}

                def hook_fn(module, input, output):
                    if hasattr(module, 'lif_module'):
                        membrane_potentials[module] = module.lif_module.membrane_potential.clone()

                # 注册钩子
                hooks = []
                for name, module in model.named_modules():
                    if isinstance(module, QConv2dLIF):
                        hooks.append(module.register_forward_hook(hook_fn))

                # 前向传播
                output = model(images)

                # 移除钩子
                for hook in hooks:
                    hook.remove()

                # 存储这个时间步的膜电位
                for idx, (module, membrane_potential) in enumerate(membrane_potentials.items()):
                    if idx not in all_membrane_potentials:
                        all_membrane_potentials[idx] = []
                    all_membrane_potentials[idx].append(membrane_potential.detach().cpu().numpy())

            # 可视化每一层的膜电位随时间的变化
            for layer_idx, layer_potentials in all_membrane_potentials.items():
                # 1. 直方图随时间的变化
                plt.figure(figsize=(15, 10))
                for t, mp in enumerate(layer_potentials):
                    plt.hist(mp.flatten(), bins=50, alpha=0.5, label=f'Step {t+1}')
                plt.title(f'Membrane Potential Distribution Over Time\nLayer {layer_idx}')
                plt.xlabel('Membrane Potential')
                plt.ylabel('Frequency')
                plt.legend()
                plt.savefig(f'membrane_visualization/layer_{layer_idx}_histogram_{quantization_setting}.png')
                plt.close()

                # 2. 热力图随时间的变化（对于卷积层）
                if len(layer_potentials[0].shape) == 4:  # 卷积层
                    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
                    fig.suptitle(f'Membrane Potential Spatial Distribution\nLayer {layer_idx}')
                    for t, mp in enumerate(layer_potentials):
                        ax = axs[t // 2, t % 2]
                        im = ax.imshow(np.mean(mp[0], axis=0), cmap='hot', interpolation='nearest')
                        ax.set_title(f'Step {t+1}')
                        fig.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(f'membrane_visualization/layer_{layer_idx}_heatmaps_{quantization_setting}.png')
                    plt.close()

                # 3. 统计信息随时间的变化
                plt.figure(figsize=(15, 10))
                means = [np.mean(mp) for mp in layer_potentials]
                stds = [np.std(mp) for mp in layer_potentials]
                plt.errorbar(range(1, time_steps+1), means, yerr=stds, fmt='o-')
                plt.title(f'Membrane Potential Statistics Over Time\nLayer {layer_idx}')
                plt.xlabel('Time Step')
                plt.ylabel('Membrane Potential')
                plt.savefig(f'membrane_visualization/layer_{layer_idx}_statistics_{quantization_setting}.png')
                plt.close()

                # 4. 膜电位随时间的变化（对某个特定神经元）
                plt.figure(figsize=(15, 10))
                if len(layer_potentials[0].shape) == 4:  # 卷积层
                    neuron_potentials = [mp[0, 0, 0, 0] for mp in layer_potentials]
                else:  # 全连接层
                    neuron_potentials = [mp[0, 0] for mp in layer_potentials]
                plt.plot(range(1, time_steps+1), neuron_potentials, 'o-')
                plt.title(f'Single Neuron Membrane Potential Over Time\nLayer {layer_idx}')
                plt.xlabel('Time Step')
                plt.ylabel('Membrane Potential')
                plt.savefig(f'membrane_visualization/layer_{layer_idx}_single_neuron_{quantization_setting}.png')
                plt.close()

    print(f"Membrane potential visualization completed for {quantization_setting}")


def main():
    torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = args_config.get_args()
    print("********** SNN Inference**********")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_dir,
        train=True,
        transform=transform_train,
        download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=10,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True)

    num_classes = 10


    predic_dic = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    trial = 4
    trial_dic = {
        0: "baseline",
        1: "2w2b",
        2: "3w3b",
        3: "4w4b"
    }
    model_path_dic = {
        0: "./figures/path_visual/final_dict_visual_baseline.pth",
        1: "./figures/path_visual/final_dict_visual_2w2b.pth",
        2: "./figures/path_visual/final_dict_visual_3w3b.pth",
        3: "./figures/path_visual/final_dict_visual_4w4b.pth",
    }
    for i in range(trial):

        model = torch.load(model_path_dic.get(i), map_location=device)
        model.to(device)
        model.eval()

        # 可视化权重（不需要数据流过）
        # visualize_weights(model, quantization_setting=trial_dic.get(i))

        # 可视化膜电位（需要数据流过）
        visualize_membrane_potential(model, test_data_loader, quantization_setting=trial_dic.get(i), time_steps=args.T)


if __name__ == '__main__':
    main()
