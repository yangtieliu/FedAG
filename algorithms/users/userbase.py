import copy
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from utils.model_config import RUNCONFIGS
from utils.model_utils import get_dataset_name
from algorithms.optimizers.optimizer import vanilla_optimizer

class User:
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        if args.device == 'cuda':
            self.model = self.model.cuda()
        self.cuda = args.device == 'cuda'
        self.id = id
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.leaning_rate = args.learning_rate
        self.local_epochs = args.local_epochs
        self.K = args.K # 本地一轮中的迭代次数（应当对应batch数量）
        self.algorithm = args.algorithm
        self.priority = 1
        self.delta_w = 0
        self.dataset = args.dataset

        # if self.train_samples > self.batch_size:
        if len(train_data) > self.batch_size:
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            self.train_loader = DataLoader(train_data, batch_size=self.train_samples, shuffle=True, drop_last=False)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loaderfull = DataLoader(test_data, batch_size=self.test_samples, shuffle=False, drop_last=False)
        self.train_loaderfull = DataLoader(train_data, batch_size=self.train_samples, shuffle=True, drop_last=False)
        self.iter_trainloader = iter(self.train_loader)
        self.iter_testloader = iter(self.test_loader)
        dataset_name = get_dataset_name(self.dataset) #小写数据集名称
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.local_model = copy.deepcopy(list(self.model.parameters()))# 此处用于个性化模型预留
        self.init_loss_fn()
        if use_adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.leaning_rate,
                                              betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
        else:
            # 此处为原始随机梯度下降
            self.optimizer = vanilla_optimizer(self.model.parameters(), lr=self.leaning_rate)
            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.leaning_rate, momentum=0.9,
            #                                 weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                   gamma=0.99) # fedftg中设置为0.998//0.99
        self.label_counts = {}

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        # "batchmean"来自FedGen
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters(self, model, beta=1): # 按beta值保留多少模型参数
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(),
                                                     self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = (1 - beta) * old_param.data.clone() + beta * new_param.data.clone()
                local_param.data =(1 - beta) * local_param.data.clone() + beta * new_param.data.clone()

    def set_shared_parameters(self, model, mode="decode_fc2"):
        for old_param, new_param in zip(self.model.get_parameters_by_keyword(mode),
                                        model.get_parameters_by_keyword(mode)):
            old_param.data = new_param.data.clone()
            # 用于指定共享层的参数更新

    def clone_model_parameters(self, param, clone_param):
        """克隆模型参数，用于个性化训练"""
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone() # clone()是创建一个新对象
            return clone_param
            # 参考FedGen源码补充
            # clone_param.data.copy_(param.data) copy_()是就地操作

    def update_parameters(self, new_params, keyword='all'):
        """占位函数，用于个性化模型，并未写完"""
        for param, new_params in zip(self.model.parameters(), new_params):
            param.data = new_params.data.clone()

    def test(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        with torch.no_grad():
            for x, y in self.test_loaderfull:
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                #if self.model_name == 'resnet18':
                    #output = self.model(x)
                    #output = nn.functional.log_softmax(output, dim=1)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
        return test_acc, loss, y.shape[0]



    def get_next_train_batch(self, count_labels=True):
        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration: #  参考FedGen补充
            # 如先前定义迭代器已耗尽则重启迭代器以获取batch
            self.iter_trainloader = iter(self.train_loader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts = torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def save_model(self):
        model_path = os.path.join('models', self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'user_{}'.format(self.id),
                                                         '.pth'))

    def load_model(self):
        model_path = os.path.join('models', self.dataset)
        self.model = torch.load(os.path.join(model_path, "server"+'.pth'))



