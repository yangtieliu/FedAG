import torch
from torchvision.datasets import CIFAR10, EMNIST, CIFAR100
from algorithms.trainmodel.resnet import resnet18
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import os
from utils.model_utils import *


def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in range(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data


class Centralized:
    def __init__(self, model, dataset):
        if model == 'cnn':
            self.model = create_model(model='cnn', dataset=dataset)[0]
        else:
            self.model = resnet18(num_classes=10, pretrained=False)
            in_channel = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_channel, 10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        #self.loss = torch.nn.NLLLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.join(current_dir, os.pardir))
        print(parent_dir)
        self.dataset = dataset
        self.root = os.path.join('/home/ytl/PycharmProjects/FedAG/', 'data', dataset, 'data')

    def test(self, test_loader):
        test_acc = 0
        y_total = 0
        self.model.eval()
        for test_x, test_y in test_loader:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            y_pred = self.model(test_x)['output']
            test_acc += (torch.sum(torch.argmax(y_pred, dim=1) == test_y)).item()
            y_total += len(test_y)
        print(test_acc)
        test_acc = test_acc / y_total
        return test_acc


    def train(self):
        epoch = 200
        batch_size = 64
        split = 'letters'
        CUDA_LAUNCH_BLOCKING=1
        if 'cifar' in self.dataset.lower():
            transform_train = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.Normalize((0.491, 0.482, 0.447),
                                      (0.247, 0.243, 0.262))]
            )
            transform_test = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.491, 0.482, 0.447),
                                      (0.247, 0.243, 0.262))]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]
            )
        if self.dataset.lower() == 'cifar10':
            train_dataset = CIFAR10(root=self.root, train=True, download=False, transform=transform_train)
            test_dataset = CIFAR10(root=self.root, train=False, download=False, transform=transform_test)
        elif self.dataset.lower() == 'emnist':
            train_dataset = EMNIST(root=self.root, train=True, download=False, transform=transform, split=split)
            test_dataset = EMNIST(root=self.root, train=True, download=False, transform=transform, split=split)
            if split == 'letters':
                classes = train_dataset.classes
                fliter_indices_train = [i for i, (_, label) in enumerate(train_dataset) if label != 0]
                fliter_dataset_train = torch.utils.data.Subset(train_dataset, fliter_indices_train)
                train_dataset = [(sample, label-1) for sample, label in fliter_dataset_train]
                #train_dataset = torch.utils.data.Dataset(train_dataset)
                fliter_indices_test = [i for i, (_, label) in enumerate(test_dataset) if label != 0]
                fliter_dataset_test = torch.utils.data.Subset(test_dataset, fliter_indices_test)
                test_dataset = [(sample, label - 1) for sample, label in fliter_dataset_test]
                #test_dataset = torch.utils.data.Dataset(test_dataset)
        elif self.dataset.lower() == 'cifar100':
            train_dataset = CIFAR100(root=self.root, train=True, download=False, transform=transform)
            test_dataset = CIFAR100(root=self.root, train=False, download=False, transform=transform)
        else:
            print("{} dataset is not in training plan".format(self.dataset))
            return
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.cuda()
        self.model.train()
        num_train_data = len(train_dataset)
        num_test_data = len(test_dataset)
        for i in range(epoch):
            trn_iter = iter(train_dataloader)
            for _ in range(int(np.ceil(num_train_data / batch_size))):
                self.optimizer.zero_grad()
                (train_x, train_y) = next(trn_iter)
                #print(train_x.shape, train_y.shape)
                train_x, train_y = train_x.cuda(), train_y.cuda()
                y_pred = self.model(train_x)['logit']
                #print(y_pred.shape)
                pred_loss = self.loss(y_pred, train_y.reshape(-1).long())
                pred_loss = pred_loss / list(train_y.size())[0]
                # print(pred_loss)
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
            test_acc = self.test(test_dataloader)
            print('epoch', i, test_acc)
            self.lr_scheduler.step()


trainer = Centralized(model='resnet18', dataset='CIFAR10')
trainer.train()

