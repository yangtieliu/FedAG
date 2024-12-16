import json
import numpy as np
import os
import torch
from algorithms.trainmodel.model import Net
from torch.utils.data import DataLoader
from algorithms.trainmodel.generator import Generator
from algorithms.trainmodel.resnet import resnet18
from datetime import datetime
from torchvision import models

METIRCS = ['glob_acc', 'per_acc', 'server_acc', 'glob_loss', 'per_loss', 'server_loss',
           'user_train_time', 'server_agg_time', 'server_training_time']

def get_data_dir(dataset):
    if 'emnist' in dataset.lower():
        # dataset 输入格式为：EMNIST-alpha0.1-ratio0.5-user20-0-letters
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').replace('user', '').split('-')
        alpha, ratio, user = dataset_[1], dataset_[2], dataset_[3]
        class_number = 26
        types = 'letters' # 即只使用EMnist数据集中的letters相关样本
        path_prefix = os.path.join('data', 'EMNIST', f'u{user}c{class_number}-{types}-alpha{alpha}-ratio{ratio}')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
    elif 'Mnist' in dataset:
        # dataset 输入格式为：Mnist-alpha0.1-ratio0.5-user20
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').replace('user', '').split('-')
        alpha, ratio, user = dataset_[1], dataset_[2], dataset_[3]
        class_number = 10
        path_prefix = os.path.join('data', 'Mnist', f'u{user}c{class_number}-alpha{alpha}-ratio{ratio}')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
    elif 'cifar10' in dataset.lower():
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').replace("user", "").split('-')
        alpha, ratio, user = dataset_[1], dataset_[2], dataset_[3]
        class_number = 10
        path_prefix = os.path.join('data', 'CIFAR10', f'u{user}c{class_number}-alpha{alpha}-ratio{ratio}')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
    elif 'fashion' in dataset.lower():
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').replace("user", "").split('-')
        alpha, ratio, user = dataset_[1], dataset_[2], dataset_[3]
        class_number = 10
        path_prefix = os.path.join('data', 'Fashion-Mnist', f'u{user}c{class_number}-alpha{alpha}-ratio{ratio}')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return train_data_dir, test_data_dir, class_number


def get_time():
    current_time = datetime.now()
    time = (f'today_{current_time.year}-{current_time.month}-{current_time.day}_time_'
            f'{current_time.hour}-{current_time.minute}-{current_time.second}')
    return time


def read_data(dataset):
    train_data_dir, test_data_dir, class_number = get_data_dir(dataset)
    clients = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pt')]
    if len(train_files) == 0:
        print(f"Dataset {dataset} not found or data type is not pt, please check the data path")
    for f in train_files:
        with open(os.path.join(train_data_dir, f), 'rb') as inf:
            cdata = torch.load(inf)
    clients.append(cdata['users'])
    train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pt')]
    for f in test_files:
        with open(os.path.join(test_data_dir, f), 'rb') as inf:
            cdata = torch.load(inf)
    test_data.update(cdata['user_data'])

    return clients, train_data, test_data

def read_user_data(index, data, count_labels=False):
    # 此处data即为read_data()函数返回的data
    id = data[0][index]
    train_data = data[1][id]
    test_data = data[2][id]
    X_train, y_train = (torch.Tensor(train_data['x']).type(torch.float32),
                        torch.Tensor(train_data['y']).type(torch.int64))
    train_data =[(X, y) for X, y in zip(X_train, y_train)]
    X_test, y_test = (torch.Tensor(test_data['x']).type(torch.float32),
                        torch.Tensor(test_data['y']).type(torch.int64))
    test_data = [(X, y) for X, y in zip(X_test, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts = torch.unique(y_train, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy() # 有必要吗？
        label_info['labels'] = unique_y # 升序排列
        label_info['counts'] = counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data

def aggregate_data_(clients, dataset, batch_size, dataset_name='train'):
    combined = []
    unique_labels = []
    class_number = 10  # 根据数据集情况更改
    label_counts = [0] * class_number  # 创建一个数据集类别数的全0列表
    total_label_counts = [0] * class_number
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y = torch.Tensor(user_data['x']).type(torch.float32), torch.Tensor(user_data['y']).type(torch.int64)
        combined += [(x, y) for x, y in zip(X, y)]
        client_unique_labels = list(torch.unique(y).detach().numpy())
        unique_labels += list(torch.unique(y).detach().numpy())
        _, label_count = torch.unique(y, return_counts=True) # 返回一个客户端的类别计数
        for label_index, label in enumerate(client_unique_labels):
            label_counts[label] = label_count[label_index].item()
            # 由于label_count为张量 ，提取其中元素使用item()以避免元素类型为张量
        # 列表元素相加方法
        total_label_counts = [a + b for a, b in zip(total_label_counts, label_counts)]
        # print(label_counts)
    # print(total_label_counts)
    if dataset_name == 'train':
        data_loader = DataLoader(combined, batch_size=batch_size, shuffle=True)
    elif dataset_name == 'test':
        data_loader = DataLoader(combined, batch_size=len(combined), shuffle=True)
    iter_loader = iter(data_loader)
    return data_loader, iter_loader, unique_labels, total_label_counts, len(combined)
def aggregate_user_data(data, batch_size):
    clients, loaded_data = data[0], data[1]
    data_loader, data_iter, unique_labels, label_counts, _ = aggregate_data_(clients, loaded_data, batch_size,
                                                                             dataset_name='train')
    # print(label_counts)
    return data_loader, data_iter, np.unique(unique_labels), np.array(label_counts)

def aggregate_user_test_data(data, batch_size):
    clients, loaded_data = data[0], data[2]
    # batch_size应为测试集样本总数
    data_loader, _,  unique_labels, _, len_data = aggregate_data_(clients, loaded_data, batch_size=batch_size,
                                                                  dataset_name='test')
    return data_loader, np.unique(unique_labels), len_data

def get_dataset_name(dataset):
    dataset = dataset.lower()
    #print(dataset)
    if 'emnist' in dataset:
        passed_dataset = 'emnist'
    elif 'mnist' in dataset:
        passed_dataset = 'mnist'
    elif 'cifar10' in dataset:
        passed_dataset = 'CIFAR10'
    elif 'fashion' in dataset:
        passed_dataset = 'fashion_mnist'
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return passed_dataset

def create_generator(dataset, algorithm='', embedding=False):
    passed_dataset = get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['FedGen', 'FedAG', 'FedFTG']])
    if 'FedGen' in algorithm:
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset += '-' + gen_model
        elif '-gen' in algorithm:
            passed_dataset += '-cnn1'
    return Generator(passed_dataset, algorithm, embedding=embedding)


def get_log_path(args, algorithm, mode, seed, time, gen_batch_size=32):
    alg = args.dataset + '-' + algorithm
    alg += '-' + mode # 生成器模型聚合方式
    alg += '-lr' +str(args.learning_rate) + '-num' + str(args.num_users) + '-bs' + str(args.batch_size)
    alg += '-epoch' + str(args.local_epochs) + '-model' + str(args.model) + '-seed' + str(seed)
    if 'FedGen' or 'FedAG' or 'DaFKD' or 'FedFTG' in algorithm:
        if int(gen_batch_size) != args.batch_size:
           alg += '-gb' + str(gen_batch_size)
    alg += '-' + str(time)
    return alg

def create_model(model, dataset):
    passed_dataset = get_dataset_name(dataset)
    if model == "cnn":
        model = Net(passed_dataset, model), model
    elif model == "resnet18":
        # num_classes需随数据集更改
        model = resnet18(num_classes=10, pretrained=False), model
        # model = models.resnet18(), model#weights='pretrained')
    else:
        raise ValueError("not supported model")
    return model
