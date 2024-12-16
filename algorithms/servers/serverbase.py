import torch
import torch.nn as nn
import numpy as np
import os
import copy
import h5py
import torch.nn.functional as F
from utils.model_utils import METIRCS, get_log_path, aggregate_user_data, get_dataset_name, get_time, aggregate_user_test_data
from utils.model_config import RUNCONFIGS


class Server:
    def __init__(self, args, model, seed):
        self.dataset = args.dataset
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        if args.device == 'cuda':
            self.model = self.model.cuda()
            self.cuda = True
        self.num_glob_iters = args.num_glob_iters # 全局迭代次数
        self.local_epochs = args.local_epochs
        self.K = args.K
        # 服务器模型迭代次数
        self.server_K = 1
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.users = []
        self.selected_users = []
        self.priority = []
        # self.num_users = len(self.users)
        self.num_users = args.num_users
        self.algorithm = args.algorithm
        #self.personalized = args.personalized # 个性化训练是否启用
        self.mode = args.mode # 生成器更新模式 #fedgen中为分类器模型参数聚合方式
        self.seed = seed
        self.lamda = args.lamda  # Fedprox正则项参数
        self.pri_alpha = 0.5
        self.pri_beta = -1
        self.pri_gamma = 0.5
        self.metrics = {key: [] for key in METIRCS}
        self.save_path = args.result_path
        self.time = get_time()  # 静态函数
        os.system("mkdir -p {}".format(self.save_path))

    def init_ensemble_configs(self, algorithm):
        dataset_name = get_dataset_name(self.dataset)
        if 'FedGen' in algorithm:
            self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)  # get(键，初始值)
            self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
            self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
            self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
            self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
            self.n_teacher_iters = 5  # 生成器模型的教师迭代次数
            self.n_student_iters = 1  # 生成器模型的学生迭代次数
            print("ensemble_lr:{}".format(self.ensemble_lr))
            print("ensemble_batch_size:{}".format(self.ensemble_batch_size))
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generator_epoches = RUNCONFIGS[dataset_name].get('generator_epochs', 10)
        self.generator_K = RUNCONFIGS[dataset_name].get('generator_K', 5)  # 服务器生成器一个epoch的迭代次数 cifar10为1
        self.generator_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.generator_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        print("unique labels:{}".format(self.unique_labels))

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            users = self.selected_users
        for user in users:
            if not self.algorithm == 'FedGen':
                user.set_parameters(self.model, beta=beta)
            else:
                if mode == 'all':
                    user.set_parameters(self.model, beta=beta)
                else:
                    user.set_shared_parameters(self.model, mode=mode)

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data += ratio * user_param.data.clone()
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self, partial):
        if partial:
            for param in self.model.get_parameters_by_keyword(keyword=self.mode):
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                # 清空服务器模型参数
                param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, ratio=user.train_samples / total_train, partial=partial)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server"+".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server"+".pt")
        assert os.path.exists(model_path)
        self.model = torch.load(model_path)

    def calculate_priority(self, users, num_users):
        # 返回优先级最高的前指定数量个的客户端编号及对应优先级值
        self.priority_idx = []
        for user in users:
            self.priority.append(user.priority)
        self.priority_idx = np.argsort(-np.array(self.priority))
        self.priority_idx = self.priority_idx[:num_users]
        self.priority = [self.priority[i] for i in self.priority_idx]
        return self.priority_idx, self.priority

    def get_client_priority(self, users):
        """
        计算所有客户端的优先级
        :param users:
        :return:
        """
        users = self.users # 客户端模型
        test_acc, test_losses = self.evaluate(save=False, selected=False)
        for i, user in enumerate(users):
            # test_acc_priority = test_acc[i] ##test_acc_priority = test_acc[i]-min(test_acc)
            # test_acc_priority = test_acc[i]-np.average(test_acc)
            # 保留归一化的处理
            user.priority = (test_acc[i]) * self.pri_alpha + test_losses[i]/sum(test_losses) * self.pri_beta + user.delta_w * self.pri_gamma
            user.priority = user.priority.item()
            print("Client {} Priority are {}".format(i+1, user.priority))
            print("test_acc_weight: {}, test_loss_weight: {}, delta_wk : {}".
                  format(test_acc[i]/sum(test_acc), test_losses[i]/sum(test_losses), user.delta_w))
        # 计算所有客户端的优先级
        # 保留是否需要归一化的可能性

    def select_users(self, num_users, way, return_idx=False):
        if num_users == len(self.users):
            print("all users are selected")
            return self.users
        num_users = min(num_users, len(self.users))
        if return_idx:
            if way == 'random':
                user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)
                return [self.users[i] for i in user_idxs], user_idxs
            else:
                # 根据优先级，返回优先级最高的前num_users个客户端对象及编号及对应优先级值
                user_idxs, _ = self.calculate_priority(self.users, num_users)
                print(user_idxs)
                return [self.users[i] for i in user_idxs], user_idxs
        else:
            if way == 'random':
                return np.random.choice(self.users, num_users, replace=False)
            else:
                user_idxs, _ = self.calculate_priority(self.users, num_users)
                return [self.users[i] for i in user_idxs]

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.ensemble_loss = nn.KLDivLoss()# reduction = "batchmean" 默认参数
        self.ce_loss = nn.CrossEntropyLoss()

    def save_results(self, args):
        alg = get_log_path(args, self.algorithm, self.mode, self.seed, self.time, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), mode='w') as hf:
            for key in self.metrics:
                #print("key device:", key)
                #print(self.metrics[key].device if not type(self.metrics[key]) == list else 'cpu')
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()

    def test(self, selected=False):
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test() #correct_samples, c_loss, num_samples(指测试集样本数量）
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, losses

    def evaluate(self, save=True, selected=False):
        test_acc = []
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = np.sum(test_accs)*1.0 / np.sum(test_samples)
        glob_loss = np.sum([x*y.cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item()/np.sum(test_samples)
        for i in range(len(test_ids)):
            if '{}_acc'.format(test_ids[i]) not in self.metrics:
                self.metrics['{}_acc'.format(test_ids[i])] = []
                self.metrics['{}_loss'.format(test_ids[i])] = []
            # 记录各客户端的百分比精度和损失值
            test_acc.append(test_accs[i]/test_samples[i])
            self.metrics['{}_acc'.format(test_ids[i])].append(test_accs[i] / test_samples[i])
            self.metrics['{}_loss'.format(test_ids[i])].append(test_losses[i].cpu().detach())

        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy = {:.4f}, Test Loss = {:.2f}".format(glob_acc, glob_loss))
        return test_acc, test_losses

    def evaluate_server(self, data, save=True):
        """
        用于测试服务器模型的性能
        :param model: 受测服务器模型
        :param data: 所有客户端的测试集之和
        :param save: 选择是否保存测试结果
        :return: 返回测试结果
        """
        self.model.eval()
        test_acc = 0
        test_loss = 0
        test_dataloader, unique_labels, len_data = aggregate_user_test_data(data, batch_size=len(data[2]))
        # self.selected_users[0].train_loaderfull, self.selected_users[0].unique_labels, self.selected_users[0].train_samples
        #aggregate_user_data(data, batch_size=len(data[1]))
            #self.selected_users[0].test_loaderfull, self.selected_users[0].unique_labels, self.selected_users[0].test_samples
        with torch.no_grad():  # 用于不追踪梯度
            for x, y in test_dataloader:
                if self.cuda:
                    x, y = x.cuda(), y.cuda()
                #if self.model_name == 'resnet18':
                    #output = self.model(x)
                    #output = nn.functional.log_softmax(output, dim=1)
                #else:
                output = self.model(x)['output']
                test_loss += self.loss(output, y).item()
                test_acc += (torch.sum(torch.argmax(output, 1) == y)).item()
                # print(test_acc)
        if save:
            print(test_acc)
            print(len_data)
            self.metrics['server_acc'].append(test_acc / len_data)
            self.metrics['server_loss'].append(test_loss)#.cpu().detach().numpy()
        print("Server model Accuracy:{:.4f}".format(test_acc / len_data))
        return test_acc / len_data, test_loss, len_data







