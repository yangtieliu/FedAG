import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR


# 初始化相关组件
class ResNetTrainer:
    def __init__(self, root, num_epochs=50, batch_size=32, lr=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        # pretrained参数已不再使用，使用预训练模型需指定weights参数为"pretrained"
        self.model = resnet18(pretrained=False, num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        #self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.98)

        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.1994, 0.2023, 0.2010]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        # 加载数据集
        self.train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
        self.test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform_test)

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    def train_one_epoch(self):
        self.model.train()

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(self.test_dataset)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def train_and_evaluate(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
            print("epoch {}".format(epoch))
            self.evaluate()
            self.scheduler.step()


# 使用示例
trainer = ResNetTrainer('/home/ytl/PycharmProjects/FedAG/data/CIFAR10/data', num_epochs=50)
trainer.train_and_evaluate()
