from tqdm import trange
import torch
import random
import numpy as np
import os
import argparse
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

random.seed(42)
np.random.seed(42)


def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in range(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data


def get_dataset(mode='train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # normalize((0.5,),(0.5,))
    ])
    dataset = MNIST(root='./data', train=True if mode == 'Train' else False, download=True, transform=transform)
    n_sample = len(dataset.data)  # 数据集样本数
    SRC_N_CLASS = len(dataset.classes)  # 类别数
    train_loader = DataLoader(dataset, batch_size=n_sample, shuffle=False)
    print("Loading data ........")
    # for循环冗余代码，可删除
    # （由于dataloader中batch_size即为样本集大小，故使用本循环对dataset.data/targets无影响
    for _, xy in enumerate(train_loader, 0):  # 0表示迭代起始位置
        dataset.data, dataset.targets = xy
    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(dataset.data.cpu().detach().numpy(),
                                            dataset.targets.cpu().detach().numpy(),
                                            SRC_N_CLASS)
    print(f'{mode.upper()} SET:\n Total #samples: {n_sample}. sample.shape:{dataset.data[0].shape}')
    print("# samples per class:\n", [len(v) for v in data_by_class])
    return data_by_class, n_sample, SRC_N_CLASS


def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sample_ratio=0.5):
    min_sample = 10  # 最少总样本数
    min_size = 0  # 客户端最小样本数
    # 执行采样
    while min_size <= min_sample:
        print('Trying to find avaliable data separation')
        idx_batch = [{} for _ in range(NUM_USERS)]
        sample_per_user = [0 for _ in range(NUM_USERS)]  # 每个客户端的样本数
        # 每个客户端的最大样本数
        max_sample_per_user = sample_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sample_ratio < 1:
                sample_for_l = int(min(max_sample_per_user, int(sample_ratio * len(data[l]))))
                # 根据采样率获得指定数量的第l类的样本
                idx_l = idx_l[:sample_for_l]
                print(f'Class {l} sample size: {len(idx_l)}/{len(data[l])}')
            # 获取每个客户端分配样本的比例
            propotions = np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            propotions = np.array([p * (n_per_user < max_sample_per_user)
                                   for p, n_per_user in zip(propotions, sample_per_user)])
            propotions = propotions / propotions.sum()
            propotions = (np.cumsum(propotions * len(idx_l))).astype(int)[:-1]
            for u, new_idx in enumerate(np.split(idx_l, propotions)):
                idx_batch[u][l] = new_idx.tolist()
                sample_per_user[u] += len(idx_batch[u][l])  # len(new_idx)效果一样
        min_size = min(sample_per_user)

    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels = [set() for _ in range(NUM_USERS)]  # 统计各客户端的样本类别
    print('Processing users...')
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0:
                continue
            X[u].extend(data[l][indices].tolist())
            y[u].extend([l] * len(indices))  # (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)
    return X, y, Labels, idx_batch, sample_per_user


def devide_test_data(NUM_USERS, test_data, SRC_CLASSES, Labels, unknown_test):
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}  # 记录每个类别的测试集总样本数
    for user in trange(NUM_USERS):
        if unknown_test:
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels = list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS)
            # 断言用于每个用户的测试集样本数不超过总测试集样本数
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user].extend(test_data[l][idx[l]:idx[l] + num_samples].tolist())
            test_y[user].extend([l] * num_samples)
            idx[l] += num_samples
    return test_X, test_y


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--format', type=str, default='pt', choices=['pt', 'json'])
        parser.add_argument('--n_class', type=int, default=10, help='number of class')
        parser.add_argument('--min_sample', type=int, default=10, help='min number samples of per user')
        parser.add_argument('--sampling_ratio', type=float, default=0.5, help='ratio of sampling training samples')
        parser.add_argument('--alpha', type=float, default=0.5, help='alpha of dirichlet distribution')
        parser.add_argument('--unknown_test', type=int, default=0, help='whether allow users can test unseen label')
        parser.add_argument('--n_user', type=int, default=10, help='number of users')
        args = parser.parse_args()
        print()
        print(f'Generating NIID Dirichlet dataset with args:\n'
              f'Number of users:{args.n_user}\n'
              f'Number of classes:{args.n_class}\n'
              f'min samples per user:{args.min_sample}\n'
              f'Alpha of dirichlet distribution:{args.alpha}\n'
              f'Ratio of sampling training samples:{args.sampling_ratio}\n')
        NUM_USERS = args.n_user

        path_prefix = f'u{args.n_user}c{args.n_class}-alpha{args.alpha}-ratio{args.sampling_ratio}'

        def process_user_data(mode, data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0):
            # 用于存储划分后的数据集，并保存用于绘制数据集的标签分布的相关信息
            user_data_count = [{"train_idx_batch_length": np.zeros(len(SRC_CLASSES)).astype(int),
                                "train_samples_per_user": 0} for _ in range(NUM_USERS)]
            if mode == 'train':
                X, y, Labels, idx_batch, samples_per_user = devide_train_data(
                    data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha, args.sampling_ratio)
            if mode == 'test':
                X, y = devide_test_data(NUM_USERS, data, SRC_CLASSES, Labels, unknown_test)
            dataset = {'user': [], 'user_data': {}, 'num_samples': []}
            for u in range(NUM_USERS):
                uname = 'f_{0:05d}'.format(u)
                dataset['users'].append(uname)
                dataset['user_data'][uname] = {'x': torch.tensor(X[u], dtype=torch.float32),
                                               'y': torch.tensor(y[u], dtype=torch.int64)}
                dataset['num_samples'].append(len(X[u]))  # 对于train模式来说，append(samples_per_user[u])效果一样
            print("{} sample by user:".format(mode.upper()), dataset['num_samples'])

            # 划分后数据集存储路径
            data_path = f'./{path_prefix}/{mode}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            # 存储路径的文件格式及名称
            data_path = os.path.join(data_path, "{}.".format(mode) + args.format)
            if args.format == 'json':
                raise NotImplementedError(
                    'json is not supported because tensor train/test data cannot be stored as json')
            elif args.format == 'pt':
                with open(data_path, 'wb') as f:
                    print(f'Dumping data to {data_path}...')
                    torch.save(dataset, f)
            # 输出各客户端的训练集信息
            if mode == 'train':
                for u in range(NUM_USERS):
                    print('{} samples in total'.format(samples_per_user[u]))
                    train_info = ''
                    n_samples_for_u = 0
                    for l in sorted(list(Labels[u])):
                        n_samples_for_l = len(idx_batch[u][l])
                        n_samples_for_u += n_samples_for_l
                        user_data_count[u]["train_idx_batch_length"][l] = n_samples_for_l
                        train_info += "c={},n={}|".format(l, n_samples_for_l)
                    user_data_count[u]['train_samples_per_user'] = n_samples_for_u
                    data_count_path = f'./{path_prefix}/count'
                    if not os.path.exists(data_count_path):
                        os.mkdir(data_count_path)
                    data_count_path = os.path.join(data_count_path, "count." + args.format)
                    with open(data_count_path, 'wb') as f:
                        print(f'Dumping data count to {data_count_path}...')
                        torch.save(user_data_count, f)
                    print(train_info)
                    print(
                        "{} Labels/{} Number of training samples for user[{}]:".format(len(Labels[u]), n_samples_for_u,
                                                                                       u)
                    )
                # 返回类别种类，每个客户端的对应类别的样本索引，每个客户端的样本数
                return Labels, idx_batch, samples_per_user

        print(f"Reading source dataset.")
        train_data, n_train_sample, SRC_N_CLASS = get_dataset(mode='train')
        test_data, n_test_sample, SRC_N_CLASS = get_dataset(mode='test')
        SRC_CLASSES = [l for l in range(SRC_N_CLASS)]
        random.shuffle(SRC_CLASSES)
        print("{} labels in total".format(len(SRC_CLASSES)))
        Labels, idx_batch, samples_per_user = process_user_data('train', train_data, n_train_sample, SRC_CLASSES,
                                                                unknown_test=args.unknown_test)
        process_user_data('test', test_data, n_test_sample, SRC_CLASSES, Labels, unknown_test=args.unknown_test)
        print("Finish Generating users samples")
    except Exception as e:
        print("Error Occurred : {}".format(e))
        exit(1)

if __name__ == "__main__":
    main()
