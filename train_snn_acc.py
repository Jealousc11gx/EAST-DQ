import dill
from quant_net import *
from training_utils import *
import math
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
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
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True)

        num_classes = 10

    criterion = nn.CrossEntropyLoss()

    model = Q_ShareScale_VGG8(args.T, args.dataset).cuda()

    # print(model)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    else:
        print("Current does not support other optimizers other than sgd or adam.")
        exit()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)  # 同一个epoch中的batch的lr是一样的

    ########################################################

    best_accuracy = 0
    dir = f'./logs/{args.arch}/{args.dataset}/T4/4w4b_hard_IF_125epoch'
    writer = SummaryWriter(log_dir=dir)
    print(f'this test\'s result is in{dir}')
    # tracemalloc.start()

    ###########################################################

    for epoch_ in range(args.epoch):
        loss = 0
        accuracy = 0
        loss = train(args, train_data_loader, model, criterion, optimizer, epoch_)
        accuracy = test(model, test_data_loader, criterion)

        writer.add_scalar('Loss/train', loss, epoch_)
        writer.add_scalar('Accuracy/test', accuracy, epoch_)

        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            checkdir(f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch")
            torch.save(model,
                       f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch/final_dict_4w4b_hard_IF.pth")

        if (epoch_ + 1) % args.test_display_freq == 0:
            print(
                f'Train Epoch: {epoch_}/{args.epoch} Loss: {loss:.6f} Accuracy: {accuracy:.3f}% Best Accuracy: {best_accuracy:.3f}%')
        if epoch_ == 24 or epoch_ == 49 or epoch_ == 74 or epoch_ == 99:
            torch.save(model, f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch/{epoch_}_dict_4w4b_hard_IF.pth")
            print(f"model saved in epoch {epoch_}")


    writer.close()


def train(args, train_data, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()

        output = model(imgs)

        train_loss = sum([criterion(s, targets) for s in output]) / args.T
        #  对于图片需要做重复 所以这里使用的是一段时间之内的平均损失

        train_loss.backward()
        optimizer.step()

    return train_loss.item()


if __name__ == '__main__':
    main()
