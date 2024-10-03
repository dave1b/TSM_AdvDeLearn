#
# Exercise 02 for advanced deep learning course
#
from datetime import datetime

#
# Construct a deep CNN model for Pet Classification
#


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
import matplotlib.pyplot as plt
from torchvision import transforms

import wandb
from mpl_toolkits.mplot3d.proj3d import transform
from scipy.stats import triang
# from torch.xpu import device
from torchinfo import summary
from torcheval.metrics import MulticlassAccuracy

import numpy as np

img_size = 500
num_classes = 37


def get_data_set(batch_size):
    #
    # CenterCrop is one possibility, but you can also try to resize the image
    #
    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
         transforms.RandomGrayscale(p=0.05), torchvision.transforms.RandomRotation(degrees=20),
         torchvision.transforms.CenterCrop(img_size)])
    data_train = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', download=True, transform=transform)
    data_test = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', split='test', download=True, transform=transform)

    len_train = (int)(0.8 * len(data_train))
    len_val = len(data_train) - len_train

    data_train_subset, data_val_subset = torch.utils.data.random_split(data_train, [len_train, len_val])

    data_train_loader = torch.utils.data.DataLoader(dataset=data_train_subset, shuffle=True, batch_size=batch_size)
    data_val_loader = torch.utils.data.DataLoader(dataset=data_val_subset, shuffle=True, batch_size=batch_size)
    data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    print(f"Length of datasets - train:{len_train}, val:{len_val}, test:{len(data_test)}")

    return data_train_loader, data_val_loader, data_test_loader


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class Metrics:
    def __init__(self, train_accs, train_losses, val_accs, val_losses, n_epochs, n_parameters, batch_size, lr, optimizer):
        self.train_accs = train_accs
        self.train_losses = train_losses
        self.val_accs = val_accs
        self.val_losses = val_losses
        self.n_epochs = n_epochs
        self.n_parameters = n_parameters
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer

    def to_csv(self):
        timestamp = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        return [self.train_accs[-1], self.train_losses[-1], self.val_accs[-1], self.val_losses[-1], self.n_epochs,
                f"{self.n_parameters:,}".replace(",", "'"),
                self.batch_size, self.lr, timestamp]


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    run = wandb.init(project="OxfordPet", config={'epochs': num_epochs, 'batch_size': train_loader.batch_size})
    step_count = 0
    print('Start training')
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    metrics = MulticlassAccuracy(num_classes=37)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        metrics.reset()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            metrics.update(predicted, labels)
            train_acc = metrics.compute()

            if (step + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Step [{step + 1}/{total_step}], '
                      f'Loss: {loss:.4f}, '
                      f'Accuracy: {train_acc:.3f}')
            train_metrics = {'train/train_loss:': loss,
                             'train/train_acc': train_acc,
                             'train/epoch': epoch}
            step_count += 1
            wandb.log(train_metrics, step=step_count)
        model.eval()
        train_accs.append(train_acc)
        train_losses.append(loss.item())

        with torch.no_grad():
            val_loss = []
            metrics.reset()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                metrics.update(predicted, labels)
                val_loss.append(loss.item())
            val_acc = metrics.compute()
            val_loss_mean = np.mean(val_loss)

            print(f'Val Accuracy: {val_acc:.3f}, loss: {val_loss_mean:.4f}')
            val_accs.append(val_acc)
            val_losses.append(val_loss_mean)
        val_metrics = {'val/val_loss': val_loss_mean,
                       'val/val_acc': val_acc}
        # log both metrics
        wandb.log(val_metrics, step=step_count)

    wandb.finish()

    # move to cpu
    train_accs = [acc.cpu().numpy() for acc in train_accs]
    # train_losses = [loss.cpu() for loss in train_losses]
    val_accs = [acc.cpu().numpy() for acc in val_accs]

    return model, (train_accs, train_losses, val_accs, val_losses)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # test if it worked
        x = torch.ones(1, device=device)
        print('Using CUDA device')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        x = torch.ones(1, device=device)
        print('Using MPS device')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device


def evaluate_final_model(model, test_loader, device):
    metrics = MulticlassAccuracy(num_classes=37)
    model.eval()
    with torch.no_grad():
        metrics.reset()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            metrics.update(predicted, labels)
        test_acc = metrics.compute()
        print(f'Final test Accuracy: {test_acc:.2f}')


def write_results_to_csv(metric: Metrics, filename):
    with open(filename, 'r') as file:
        index = int(file.readlines()[-1].split(',')[0]) + 1
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(np.append(index, metric.to_csv()))
    return index


def save_train_curves_plots(results: Metrics, filename):
    f = plt.figure(figsize=(12, 4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.set_xlabel('Epochs')
    ax1.plot(results.train_losses, label='Training loss')
    ax1.plot(results.val_accs, label='Validation loss')
    ax1.legend()
    ax1.grid()
    ax2.set_xlabel('Epochs')
    ax2.plot(results.train_accs, label='Training acc')
    ax2.plot(results.val_accs, label='Validation acc')
    ax2.legend()
    ax2.grid()
    plt.savefig(filename)


def main():
    # wandb.login()

    batch_size = 64
    train_loader, val_loader, test_loader = get_data_set(batch_size)
    device = get_device()
    cnn = DeepCNN()
    n_parameters = sum(p.numel() for p in cnn.parameters())
    print(f"Number of parameters: {n_parameters}")
    n_epochs = 20
    lr = 0.001
    weight_decay = 0.0001
    optimizer = optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    final_model, results = train(cnn, train_loader, val_loader, loss_fn, optimizer, n_epochs, device)
    metrics = Metrics(*results, n_epochs, n_parameters, batch_size, lr, optimizer.__class__.__name__)
    index = write_results_to_csv(metrics, 'data/train_stats.csv')
    save_train_curves_plots(metrics, f'data/train_curves/train_curves_{index}.png')
    # evaluate_final_model(final_model, test_loader, device)


if __name__ == '__main__':
    main()
