import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import resnet
from utils import progress_bar


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = True

task = 'resnet50-cifar10'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(task))
        best_acc = acc

    return acc / 100


def plot_curve(train_acc_list, test_acc_list):
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title("CIFAR-10 ResNet50 Accuracy Curve")  # dataset
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend(["train", "test"])

    plt.savefig('results/{}.png'.format(task), dpi=500)


def evaluate(checkpoint):
    net.eval()
    net.load_state_dict(checkpoint['net'])

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)

            _, pred = output.topk(5, 1, largest=True, sorted=True)

            targets = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(targets).float()

            total += targets.size(0)
            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()

    print("Top 1 acc: {}%".format(correct_1.item() / total * 100.0))
    print("Top 5 acc: {}%".format(correct_5.item() / total * 100.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = resnet.resnet50()
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epochs = 200
    train_acc_list, test_acc_list = [], []
    for epoch in range(start_epoch, start_epoch+epochs):
        # Train
        train_acc = train(epoch)
        train_acc_list.append(train_acc)
        # test
        test_acc = test(epoch)
        test_acc_list.append(test_acc)

        scheduler.step()
    plot_curve(train_acc_list, test_acc_list)

    checkpoint = torch.load('./checkpoint/{}.pth'.format(task))
    evaluate(checkpoint)
