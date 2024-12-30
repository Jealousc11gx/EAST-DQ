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
            num_workers=12,
            pin_memory=True)
        # test data loader have changed
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=256,
            shuffle=True,
            drop_last=False,
            num_workers=12,
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

        # other visualization
    def plot_average_transition_vs_batch(weight_tracking, segment_name):
        os.makedirs(f'plots/average_transition_vs_batch/{segment_name}', exist_ok=True)
        
        for name, data in weight_tracking.items():
            transitions = data.get('transitions', [])
            if len(transitions) == 0:
                print(f"No transitions data for layer: {name} in segment {segment_name}")
                continue
    
            transitions = np.array(transitions)  # 将列表转换为NumPy数组，形状为 (num_batches_total, 3, 3)
            labels = ['s1->s1', 's1->s2', 's1->s3', 's2->s1', 's2->s2', 's2->s3', 's3->s1', 's3->s2', 's3->s3']
    
            # 平均三个阶段的状态转移，假设 transitions 维度为 (num_batches_total, 3, 3)
            num_stages = 3
            batches_per_stage = 10
            avg_transition_per_batch = []
    
            # 遍历所有批次并计算每个批次在三个阶段的平均状态转移
            for batch_idx in range(batches_per_stage):
                stage_transitions = []
                for stage in range(num_stages):
                    start_idx = stage * batches_per_stage
                    end_idx = start_idx + batches_per_stage
                    stage_transitions.append(transitions[start_idx:end_idx][batch_idx])
                avg_transition = np.mean(stage_transitions, axis=0)
                avg_transition_per_batch.append(avg_transition.reshape(-1))
    
            avg_transition_per_batch = np.array(avg_transition_per_batch)  # 形状为 (batches_per_stage, 9)
    
            # 保存画图数据
            plot_data = {'avg_transition_per_batch': avg_transition_per_batch, 'labels': labels}
            with open(f'plots/average_transition_vs_batch/{segment_name}/average_transition_vs_batch_{name}_data.pkl', 'wb') as f:
                pickle.dump(plot_data, f)
            
            # 绘制状态转移随batch变化的3D折线图
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            x = np.arange(batches_per_stage)  # 批次编号 (0 到 9)
            y = np.arange(len(labels))  # 状态转移类型编号 (0 到 8)
            X, Y = np.meshgrid(x, y)
            Z = avg_transition_per_batch.T  # 转置后形状为 (9, batches_per_stage)，以适应绘图
            
            # 绘制3D折线图
            for i in range(len(labels)):
                ax.plot(x, np.full(batches_per_stage, i), Z[i], label=labels[i], marker='o', linestyle='-', linewidth=1)
            
            ax.set_xlabel('Batch Number (0-9)')
            ax.set_ylabel('State Transition Type')
            ax.set_zlabel('Average Transition Percentage')
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, rotation=45, ha='right')
            ax.set_title(f'Average State Transition vs Batch - Layer {name} - Segment {segment_name}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'plots/average_transition_vs_batch/{segment_name}/average_transition_vs_batch_{name}_3d.png', dpi=300)
            plt.close()




        
    # 图2: 2D柱状图展示状态转移的平均比例
    def plot_transition_bar(weight_tracking, segment_name):
        os.makedirs(f'plots/transition_bar/{segment_name}', exist_ok=True)
        for name, data in weight_tracking.items():
            transitions = data.get('transitions', [])
            if len(transitions) == 0:
                print(f"No transitions data for layer: {name} in segment {segment_name}")
                continue
    
            transitions = np.array(transitions)  # 将列表转换为NumPy数组
            avg_transition = np.mean(transitions, axis=0).flatten()  # 计算平均状态转移
            labels = ['s1->s1', 's1->s2', 's1->s3', 's2->s1', 's2->s2', 's2->s3', 's3->s1', 's3->s2', 's3->s3']
    
            # 保存画图数据
            plot_data = {'avg_transition': avg_transition, 'labels': labels}
            with open(f'plots/transition_bar/{segment_name}/transition_bar_{name}_data.pkl', 'wb') as f:
                pickle.dump(plot_data, f)
    
            plt.figure(figsize=(12, 6))
            sns.barplot(x=labels, y=avg_transition * 100)  # 使用百分比表示
            plt.xlabel('State Transition (sij)')
            plt.ylabel('Average Transition Percentage (%)')
            plt.title(f'Average State Transition - Layer {name}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'plots/transition_bar/{segment_name}/transition_bar_{name}.png', dpi=300)
            plt.close()
    
    # 图3: 权重状态随Batch变化的图
    def plot_weight_state_vs_batch(weight_tracking, segment_name):
        os.makedirs(f'plots/weight_state_vs_batch/{segment_name}', exist_ok=True)
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
            with open(f'plots/weight_state_vs_batch/{segment_name}/weight_state_vs_batch_{name}_data.pkl', 'wb') as f:
                pickle.dump(plot_data, f)
            
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
    
            colors = plt.cm.rainbow(np.linspace(0, 1, num_weights))
            for weight_index in range(num_weights):
                x = np.arange(num_batches)
                y = np.full(num_batches, weight_index)
                z = weight_states[:, weight_index]
    
                ax.plot(x, y, z, color=colors[weight_index], marker='o', markersize=4, linestyle='-', linewidth=1)
    
            ax.set_xlabel('Batch Number', fontsize=12)
            ax.set_ylabel('Weight Index', fontsize=12)
            ax.set_zlabel('Weight State', fontsize=12)
            ax.set_yticks(range(num_weights))
            ax.set_zlim(-1.5, 1.5)
            ax.set_zticks([-1, 0, 1])
            ax.set_title(f'Weight State Changes: {name} layer - Segment {segment_name}', fontsize=14)
    
            # 设置每个batch都显示
            ax.set_xticks(range(num_batches))
            ax.set_xticklabels(range(num_batches), rotation=45, ha='right')
            plt.setp(ax.get_xticklabels(), visible=True, fontsize=8)
    
            ax.grid(True, linestyle='--', alpha=0.7)
    
            plt.tight_layout()
            plt.savefig(f'plots/weight_state_vs_batch/{segment_name}/weight_state_vs_batch_{name}.png', dpi=300, bbox_inches='tight')
            plt.close()


    # ********************************* other visualization ****************************
    model = Q_ShareScale_VGG8(args.T, args.dataset).cuda()
    weight_tracking = initialize_weight_tracking(model)

    # ******************************load pre-train model********************************

    pretrained_path = f"{os.getcwd()}/pretrain_models/temp/0_dict_2w2b_if_hard.pth"
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
                break
        if not track_flag:
            train_batch(args, imgs, targets, model, criterion, optimizer,
                        None, batch_idx, track_flag)

    for segment_name, weight_tracking_segment in weight_tracking_segments.items():


        # plot_average_transition_vs_batch(weight_tracking_segment, segment_name)  # 绘制图1
        # plot_transition_bar(weight_tracking_segment, segment_name)               # 绘制图2
        # plot_weight_state_vs_batch(weight_tracking_segment, segment_name)

        plot_transition_pie(weight_tracking_segment, segment_name)
        plot_average_transition_pie(weight_tracking_segment, segment_name)
        generate_average_transition_csv(weight_tracking_segment, segment_name)

        layer_power = calculate_average_power(weight_tracking_segment, segment_name)
        plot_power_consumption(layer_power, segment_name)
        generate_power_consumption_csv(layer_power, segment_name)

    # ********************************weights tracking*********************



def train_batch(args, imgs, targets, model, criterion, optimizer, weight_tracking, batch_idx, track_flag):
    """
    执行一个训练批次的操作。

    参数:
    - args: 包含全局配置参数的对象。
    - imgs: 一个批次的输入图像。
    - targets: 输入图像对应的目标值。
    - model: 要训练的模型。
    - criterion: 损失函数。
    - optimizer: 优化器。
    - weight_tracking: 权重跟踪对象，用于记录模型权重的变化。
    - batch_idx: 当前批次的索引。
    - track_flag: 一个布尔值，指示是否跟踪权重变化。

    返回:
    - 返回更新后的权重跟踪对象。
    """
    # 将模型设置为训练模式
    model.train()
    # 将图像和目标值移动到GPU
    imgs, targets = imgs.cuda(), targets.cuda()
    # 清除优化器的梯度
    optimizer.zero_grad()
    # 前向传播获取模型输出
    output = model(imgs)
    # 计算损失值，使用输出的平均值除以温度参数
    loss = sum([criterion(s, targets) for s in output]) / args.T
    # 反向传播计算梯度
    loss.backward()
    # 更新模型权重
    optimizer.step()
    # 更新权重跟踪对象
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
