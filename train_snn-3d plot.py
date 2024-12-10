import os
import dill
import numpy as np
from quant_net import *
from training_utils import *
import tracemalloc
import math
import gc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import random
import csv
from mpl_toolkits.mplot3d import Axes3D
import pickle


def main():
    torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = args_config.get_args()
    print("********** SNN simulation parameters **********")
    print(args)

    if args.dataset == 'cifar10':
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
            batch_size=256,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True)

        num_classes = 10

    def generate_average_transition_csv(weight_tracking, segment_name):
        os.makedirs(f'csv/average_transition/{segment_name}', exist_ok=True)
        for name, data in weight_tracking.items():
            transitions = data.get('transitions', [])
            if len(transitions) == 0:
                print(f"No transitions data for layer: {name} in segment {segment_name}")
                continue
            avg_transition = np.mean(transitions, axis=0)
            with open(f'csv/average_transition/{segment_name}/average_transition_{name}.csv', 'w',
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['From State', 'To State', 'Count'])
                for i in range(3):
                    for j in range(3):
                        writer.writerow([f's{i + 1}', f's{j + 1}', int(avg_transition[i, j])])

    def generate_power_consumption_csv(layer_power, segment_name):
        os.makedirs(f'csv/power_consumption/{segment_name}', exist_ok=True)
        with open(f'csv/power_consumption/{segment_name}/power_consumption.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'Power Consumption (fJ)'])
            for layer, power in layer_power.items():
                writer.writerow([layer, f'{power:.2f}'])

    def calculate_average_power(weight_tracking, segment_name):
        os.makedirs(f'csv/power_consumption/{segment_name}', exist_ok=True)
        power_consumption = {
            's11': 0, 's12': 5.6, 's13': 12.7,
            's21': 12.7, 's22': 0, 's23': 12.7,
            's31': 12.7, 's32': 5.6, 's33': 0
        }

        layer_power = {}
        for name, data in weight_tracking.items():
            transitions = np.array(data['transitions'])
            avg_transition = np.mean(transitions, axis=0)
            total_transitions = np.sum(avg_transition)
            transition_ratios = avg_transition / total_transitions

            average_power = sum(
                [transition_ratios[i // 3, i % 3] * power_consumption[f's{i // 3 + 1}{i % 3 + 1}'] for i in range(9)])
            layer_power[name] = average_power

        with open(f'csv/power_consumption/{segment_name}/average_power_consumption.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'Average Power Consumption (fJ)'])
            for layer, power in layer_power.items():
                writer.writerow([layer, f'{power:.2f}'])

        return layer_power

    def plot_power_consumption(layer_power, segment_name):
        os.makedirs(f'plots/power_consumption/{segment_name}', exist_ok=True)
        layers = list(layer_power.keys())
        power_values = list(layer_power.values())
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(layers)), power_values, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title(f'Average Power Consumption per Layer - Segment {segment_name}', fontsize=16)
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Power Consumption (fJ)', fontsize=14)
        plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, power in enumerate(power_values):
            plt.text(i, power, f'{power:.2f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'plots/power_consumption/{segment_name}/power_consumption.png', dpi=300, bbox_inches='tight')
        plt.close()

    def initialize_weight_tracking(model, num_weights_per_kernel=4):
        """
        初始化权重跟踪结构。

        该函数遍历模型的所有模块，当遇到QConv2dLIF模块时，层次化抽样
        这些权重的索引和变化历史将被记录在一个字典中，以便后续分析。

        参数:
            model: 模型对象，其权重将被跟踪。
            num_weights_per_kernel: 每个模块中要跟踪的权重数量，默认为20。

        返回:
            weight_tracking: 一个字典，包含每个被跟踪权重的索引、历史和转换。
        """
        weight_tracking = {}
        for name, module in model.named_modules():
            if isinstance(module, QConv2dLIF):
                de_weights = module.get_quantized_weights().detach().cpu().numpy()  # De_quantized 这里不算值 不影响 这里是反量化后的权重
                C_out, C_in, H, W = de_weights.shape
                tracking_indices = []
                # 在 initialize_weight_tracking 中确保对每个卷积核都进行均匀采样
                for c_out in range(C_out // 4):
                    for c_in in range(C_in // 4):
                        # 在 H x W 上进行均匀采样
                        h_indices = np.linspace(0, H - 1, min(H, num_weights_per_kernel // 2), dtype=int)
                        w_indices = np.linspace(0, W - 1, min(W, num_weights_per_kernel // 2), dtype=int)

                        for h in h_indices:
                            for w in w_indices:
                                tracking_indices.append((c_out, c_in, h, w))

                # 修改 initialize_weight_tracking 函数，使用 NumPy 数组进行初始化
                weight_tracking[name] = {
                    'indices': tracking_indices,
                    'history': [],
                    'transitions': []
                }

        return weight_tracking

    # ******************************weight-trans-plot************************

    def plot_transition_pie(weight_tracking, segment_name):
        os.makedirs(f'plots/transition_pie/{segment_name}', exist_ok=True)
        for name, data in weight_tracking.items():
            transitions = data.get('transitions', [])
            if len(transitions) == 0:
                print(f"No transitions data for layer: {name} in segment {segment_name}")
                continue

            fig, axs = plt.subplots(2, 5, figsize=(25, 10))
            fig.suptitle(f'Transition Pies: {name} layer - Segment {segment_name}', fontsize=16)

            for i, transition in enumerate(transitions):
                row, col = divmod(i, 5)
                ax = axs[row, col]
                labels = ['s1->s1', 's1->s2', 's1->s3',
                          's2->s1', 's2->s2', 's2->s3',
                          's3->s1', 's3->s2', 's3->s3']
                sizes = transition.flatten()
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Batch {i} to {i + 1}')
                plt.setp(texts, size=8)
                plt.setp(autotexts, size=8, weight="bold")

            # Hide unused subplots if any
            for i in range(len(transitions), 10):
                fig.delaxes(axs[i // 5, i % 5])

            plt.tight_layout()
            plt.savefig(f'plots/transition_pie/{segment_name}/combined_transition_pie_{name}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    def plot_average_transition_pie(weight_tracking, segment_name):
        os.makedirs(f'plots/average_transition_pie/{segment_name}', exist_ok=True)
        for name, data in weight_tracking.items():
            transitions = data.get('transitions', [])
            if len(transitions) == 0:
                print(f"No transitions data for layer: {name} in segment {segment_name}")
                continue
            avg_transition = np.mean(transitions, axis=0)
            total_transitions = np.sum(avg_transition)
            transition_ratios = avg_transition / total_transitions
            plt.figure(figsize=(10, 10))
            labels = ['s1->s1', 's1->s2', 's1->s3',
                      's2->s1', 's2->s2', 's2->s3',
                      's3->s1', 's3->s2', 's3->s3']
            sizes = transition_ratios.flatten()
            wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(f'Average Weight State Transitions: {name} layer - Segment {segment_name}', fontsize=16)
            plt.setp(texts, size=10)
            plt.setp(autotexts, size=10, weight="bold")
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f'plots/average_transition_pie/{segment_name}/average_transition_pie_{name}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()



    def plot_average_transition_vs_batch_combined(weight_tracking_segments, segment_names):
        os.makedirs(f'test_plots/average_transition_vs_batch/combined_comparison', exist_ok=True)

        # 遍历所有层
        for name in weight_tracking_segments[segment_names[0]]:
            fig, axs = plt.subplots(1, len(segment_names) + 1, figsize=(25, 10), subplot_kw={'projection': '3d'})
            fig.suptitle(f'Average State Transition per Batch - Layer {name}', fontsize=16)

            all_batches_transition = []

            # 遍历每个阶段，收集每个阶段的 transition 矩阵
            for i, segment_name in enumerate(segment_names):
                weight_tracking = weight_tracking_segments[segment_name]
                data = weight_tracking[name]
                transitions = data.get('transitions', [])
                if len(transitions) == 0:
                    print(f"No transitions data for layer: {name} in segment {segment_name}")
                    continue

                # 将当前阶段的所有批次的 transition 矩阵添加到 all_batches_transition 中
                all_batches_transition.append(transitions)  # transitions 形状为 (9, 3, 3)

                # 转换为 NumPy 数组，形状为 (9, 3, 3)
                transitions = np.array(transitions)

                # 展平每个批次的平均转移矩阵，得到形状为 (9, 9)
                transition_per_batch_flattened = transitions.reshape(9, -1)

                # 计算每个批次中每种状态转移的比例，并转换为百分比
                transition_percentage = transition_per_batch_flattened / transition_per_batch_flattened.sum(axis=1,
                                                                                                            keepdims=True) * 100

                # 绘制3D条形图
                ax = axs[i]
                x = np.arange(9)  # 9 个批次编号
                y = np.arange(9)  # 状态转移类型编号 (0 到 8 对应 9 种类型)
                X, Y = np.meshgrid(x, y)
                Z = transition_percentage.T  # 转置后形状为 (9, 9)，用于绘图

                labels = ['s11', 's12', 's13', 's21', 's22', 's23', 's31', 's32', 's33']
                colors = plt.cm.viridis(np.linspace(0, 1, len(y)))

                width = 0.3  # 设置条形图的宽度
                for j in range(len(y)):
                    ax.bar(x, Z[j], zs=j, zdir='y', width=width, color=colors[j], alpha=0.8)
                    ax.plot(x, np.full(len(x), j), Z[j], color=colors[j], marker='o', linestyle='--', linewidth=2)

                ax.set_xlabel('Batch Index')
                ax.set_ylabel('State Transition Type')
                ax.set_zlabel('Average Transition Proportion (%)')
                ax.set_xticks(np.arange(len(x)))
                ax.set_xticklabels(np.arange(len(x)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, rotation=45, ha='center', va='center')
                ax.set_title(f'Segment {segment_name}')
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            # 对阶段进行平均，得到形状为 (9, 3, 3)
            avg_transition_per_batch = np.mean(all_batches_transition, axis=0)

            # 展平每个批次的平均转移矩阵，得到形状为 (9, 9)
            avg_transition_per_batch_flattened = avg_transition_per_batch.reshape(9, -1)

            # 计算每个批次中每种状态转移的比例，并转换为百分比
            avg_transition_percentage = avg_transition_per_batch_flattened / avg_transition_per_batch_flattened.sum(
                axis=1, keepdims=True) * 100

            # 绘制平均的3D条形图
            ax = axs[-1]
            x = np.arange(9)  # 9 个批次编号
            y = np.arange(9)  # 状态转移类型编号 (0 到 8 对应 9 种类型)
            X, Y = np.meshgrid(x, y)
            Z = avg_transition_percentage.T  # 转置后形状为 (9, 9)，用于绘图

            colors = plt.cm.viridis(np.linspace(0, 1, len(y)))

            width = 0.3  # 设置条形图的宽度
            for j in range(len(y)):
                ax.bar(x, Z[j], zs=j, zdir='y', width=width, color=colors[j], alpha=0.8, bottom=0)
                ax.plot(x, np.full(len(x), j), Z[j], color=colors[j], marker='o', linestyle='--', linewidth=2)

            ax.set_xlabel('Batch Index')
            ax.set_ylabel('State Transition Type')
            ax.set_zlabel('Average Transition Proportion (%)')
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(np.arange(len(x)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, rotation=45, ha='center', va='center')
            ax.set_title(f'Average of All Segments')

            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.savefig(
                f'test_plots/average_transition_vs_batch/combined_comparison/average_transition_vs_batch_{name}_comparison.png',
                dpi=300)
            plt.close()


    def plot_weight_state_vs_batch(weight_tracking, segment_name):
        os.makedirs(f'test_plots/weight_state_vs_batch/{segment_name}', exist_ok=True)
        for name, data in weight_tracking.items():
            history = data.get('history', [])
            if len(history) == 0:
                print(f"No history data for layer: {name} in segment {segment_name}")
                continue

            # 随机选择10个权重进行跟踪
            num_weights = 10
            num_batches = len(history)
            selected_indices = np.random.choice(len(history[0]['quantized']), num_weights, replace=False)
            weight_states = np.array([history[batch]['quantized'][selected_indices] for batch in range(num_batches)])

            # 保留状态值为-1, 0, 1
            weight_states = np.clip(weight_states, -1, 1)

            # 保存画图数据
            plot_data = {'weight_states': weight_states, 'num_batches': num_batches, 'num_weights': num_weights}
            with open(f'test_plots/weight_state_vs_batch/{segment_name}/weight_state_vs_batch_{name}_data.pkl',
                      'wb') as f:
                pickle.dump(plot_data, f)

            # 创建图形
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')

            # 设置颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, num_weights))

            # 绘制权重状态变化图，增加线宽和标记大小
            for weight_index in range(num_weights):
                x = np.arange(num_batches)
                y = np.full(num_batches, weight_index)
                z = weight_states[:, weight_index]

                # 绘制曲线，增加虚线和更显著的标记
                ax.plot(x, y, z, color=colors[weight_index], marker='o', markersize=6, linestyle='--', linewidth=2)

            # 设置轴标签和标题
            ax.set_xlabel('Batch Number', fontsize=14, labelpad=15)
            ax.set_ylabel('Weight Index', fontsize=14, labelpad=15)
            ax.set_zlabel('Weight State', fontsize=14, labelpad=10)
            ax.set_yticks(range(num_weights))
            ax.set_zlim(-1.5, 1.5)
            ax.set_zticks([-1, 0, 1])
            ax.set_title(f'Weight State Changes: {name} layer - Segment {segment_name}', fontsize=16, pad=20)

            # 设置每个批次都显示
            ax.set_xticks(range(num_batches))
            ax.set_xticklabels(range(num_batches), rotation=45, ha='right')
            plt.setp(ax.get_xticklabels(), visible=True, fontsize=10)

            # 设置相机视角
            ax.view_init(elev=25, azim=-60)  # 设置较好的观察角度，使得图形各部分更加清晰

            # 增加网格线和整体布局
            ax.grid(True, linestyle='--', alpha=0.7)

            # 调整布局，确保没有部分被裁剪
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            # 保存图像
            plt.savefig(f'test_plots/weight_state_vs_batch/{segment_name}/weight_state_vs_batch_{name}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    def plot_weight_state_vs_batch_combine(weight_tracking_segments, segment_names):
        """
        绘制同一层在多个阶段的权重状态随批次的变化，结合子图展示。
        """
        os.makedirs(f'test_plots/weight_state_vs_batch_combined/', exist_ok=True)

        # 获取所有段中共有的层名
        layers = list(weight_tracking_segments[segment_names[0]].keys())

        # 对每个层绘制多个阶段的变化
        for name in layers:
            fig = plt.figure(figsize=(18, 10))

            # 创建多子图
            num_segments = len(segment_names)
            for i, segment_name in enumerate(segment_names):
                weight_tracking = weight_tracking_segments[segment_name]
                data = weight_tracking.get(name)
                if not data:
                    print(f"No data for layer: {name} in segment {segment_name}")
                    continue

                history = data.get('history', [])
                if len(history) == 0:
                    print(f"No history data for layer: {name} in segment {segment_name}")
                    continue

                # 随机选择10个权重进行跟踪
                num_weights = 10
                num_batches = len(history)
                selected_indices = np.random.choice(len(history[0]['quantized']), num_weights, replace=False)
                weight_states = np.array(
                    [history[batch]['quantized'][selected_indices] for batch in range(num_batches)])

                # 保留状态值为-1, 0, 1
                weight_states = np.clip(weight_states, -1, 1)

                # 保存数据到 CSV 文件
                csv_filename = f'test_plots/weight_state_vs_batch_combined/weight_state_vs_batch_{name}_{segment_name}.csv'
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Batch Index'] + [f'Weight {i}' for i in range(num_weights)])
                    for batch_idx in range(num_batches):
                        writer.writerow([batch_idx] + weight_states[batch_idx].tolist())

                # 在子图中绘制
                ax = fig.add_subplot(1, num_segments, i + 1, projection='3d')

                # 设置颜色
                colors = plt.cm.rainbow(np.linspace(0, 1, num_weights))

                # 绘制权重状态变化图，增加线宽和标记大小
                for weight_index in range(num_weights):
                    x = np.arange(num_batches)
                    y = np.full(num_batches, weight_index)
                    z = weight_states[:, weight_index]

                    # 绘制曲线，增加虚线和更显著的标记
                    ax.plot(x, y, z, color=colors[weight_index], marker='o', markersize=6, linestyle='--', linewidth=2)

                # 设置轴标签和标题
                ax.set_xlabel('Batch Number', fontsize=10, labelpad=15)
                ax.set_ylabel('Weight Index', fontsize=10, labelpad=15)
                ax.set_zlabel('Weight State', fontsize=10, labelpad=10)
                ax.set_yticks(range(num_weights))
                ax.set_zlim(-1.5, 1.5)
                ax.set_zticks([-1, 0, 1])
                ax.set_title(f'Segment {segment_name}', fontsize=12, pad=20)

                # 设置每个批次都显示
                ax.set_xticks(range(num_batches))
                ax.set_xticklabels(range(num_batches), rotation=45, ha='right')
                plt.setp(ax.get_xticklabels(), visible=True, fontsize=8)

                # 设置相机视角
                ax.view_init(elev=25, azim=-60)

                # 增加网格线
                ax.grid(True, linestyle='--', alpha=0.7)

            # 调整布局，确保没有部分被裁剪
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

            # 保存合并后的图像
            plt.savefig(f'test_plots/weight_state_vs_batch_combined/weight_state_vs_batch_combined_{name}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    # ********************************* other visualization ****************************
    model = Q_ShareScale_VGG8(args.T, args.dataset).cuda()
    weight_tracking = initialize_weight_tracking(model)

    # ******************************load pre-train model********************************

    pretrained_path = f"{os.getcwd()}/pretrain_models/temp/99_dict_2w2b_if_hard.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_model = torch.load(pretrained_path)
        if isinstance(pretrained_model, Q_ShareScale_VGG8):
            # 如果加载的是整个模型对象，我们直接复制其状态字典
            model.load_state_dict(pretrained_model.state_dict())
            print("Model loaded. First layer weights mean:", next(model.parameters()).mean().item())
        elif isinstance(pretrained_model, dict):
            # 如果加载的是状态字典，直接加载
            model.load_state_dict(pretrained_model)
            print("Model loaded. First layer weights mean:", next(model.parameters()).mean().item())
        else:
            print(f"Unexpected type of loaded model: {type(pretrained_model)}")
    else:
        print(f"Pretrained weights not found at {pretrained_path}, train from scratch.")

    # ******************************load pre-train model********************************

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    else:
        print("Current does not support other optimizers other than sgd or adam.")
        exit()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)
    best_accuracy = 0

    # 定义需要追踪的批次范围
    tracked_batches_segments = {
        'early': list(range(0, 10)),
        'middle': list(range(95, 105)),
        'last': list(range(180, 190))
    }
    # ********************************weights tracking*********************
    weight_tracking_segments = {segment: initialize_weight_tracking(model) for segment in tracked_batches_segments}

    for batch_idx, (imgs, targets) in enumerate(train_data_loader):
        track_flag = False
        for segment_name, tracked_batches in tracked_batches_segments.items():
            if batch_idx in tracked_batches:
                track_flag = True
                weight_tracking_segments[segment_name] = train_batch(args, imgs, targets, model, criterion, optimizer,
                                                                     weight_tracking_segments[segment_name], batch_idx,
                                                                     track_flag)
        # 如果已经超出了最大 tracked_batches 范围，退出循环
        if batch_idx > max(max(batches) for batches in tracked_batches_segments.values()):
            break

    for segment_name, weight_tracking_segment in weight_tracking_segments.items():
        # plot_transition_pie(weight_tracking_segment, segment_name)
        # plot_average_transition_pie(weight_tracking_segment, segment_name)
        # generate_average_transition_csv(weight_tracking_segment, segment_name)
        #
        # layer_power = calculate_average_power(weight_tracking_segment, segment_name)
        # plot_power_consumption(layer_power, segment_name)
        # generate_power_consumption_csv(layer_power, segment_name)
        plot_weight_state_vs_batch(weight_tracking_segment, segment_name)
    plot_weight_state_vs_batch_combine(weight_tracking_segments, list(tracked_batches_segments.keys()))
    plot_average_transition_vs_batch_combined(weight_tracking_segments, list(tracked_batches_segments.keys()))  # 绘制图1
    print("Visualization has been Done!")
    # ********************************weights tracking*********************


def train_batch(args, imgs, targets, model, criterion, optimizer, weight_tracking, batch_idx, track_flag):

    model.train()
    imgs, targets = imgs.cuda(), targets.cuda()
    optimizer.zero_grad()
    output = model(imgs)
    loss = sum([criterion(s, targets) for s in output]) / args.T
    loss.backward()
    optimizer.step()
    if track_flag:
        weight_tracking = update_weight_tracking(model, weight_tracking, batch_idx)
    return weight_tracking


def update_weight_tracking(model, weight_tracking, batch_idx):
    """
    更新权重跟踪信息。

    该函数遍历模型的所有模块，当遇到特定类型的卷积层（QConv2dLIF）时，
    计算该层权重的量化值，并记录这些量化值及其在不同批次中的变化情况。

    参数:
    - model: 模型对象，用于访问其内部的模块。
    - weight_tracking: 权重跟踪信息的字典，包含权重的量化历史和状态转移信息。
    - batch_idx: 当前批次的索引，用于记录权重变化的批次。

    返回:
    - weight_tracking: 更新后的权重跟踪信息字典。
    """
    segment_start = [0, 95, 180]

    # 遍历模型的所有模块，寻找特定类型的卷积层
    for name, module in model.named_modules():
        # 检查当前模块是否为 QConv2dLIF 类型且名称在指定范围内
        if isinstance(module, QConv2dLIF) and name in ['ConvLif2', 'ConvLif3', 'ConvLif4', 'ConvLif5', 'ConvLif6']:
            # 计算当前模块权重的量化值
            quantized_weights, _ = w_q_inference(module.conv_module.weight, module.num_bits_w, module.beta[0])
            quantized_weights = quantized_weights.detach().cpu()
            # 获取当前关注的权重索引
            indices = weight_tracking[name]['indices']
            quantized_states = []

            # 根据保存的索引提取量化的权重
            for (c_out, c_in, h, w) in indices:
                quantized_states.append(quantized_weights[c_out, c_in, h, w])

            # 将权重值限制在[-1, 1]范围内
            quantized_states = np.clip(quantized_states, -1, 1)
            # 将权重值转换为整数形式，以便于后续处理
            quantized_states = (np.array(quantized_states) + 1).astype(int)

            # 将量化后的权重信息和当前批次索引添加到历史记录中
            weight_tracking[name]['history'].append({
                'quantized': quantized_states,
                'batch': batch_idx
            })

            # 如果当前不在段初始序号，计算权重状态的转移情况
            if batch_idx not in segment_start:
                prev_states = weight_tracking[name]['history'][-2]['quantized']
                transition_matrix = np.zeros((3, 3), dtype=int)

                # 计算当前批次与前一个批次的状态变化情况
                for prev, curr in zip(prev_states, quantized_states):
                    transition_matrix[prev, curr] += 1
                weight_tracking[name]['transitions'].append(transition_matrix)

    # 返回更新后的权重跟踪信息
    return weight_tracking


if __name__ == '__main__':
    main()
