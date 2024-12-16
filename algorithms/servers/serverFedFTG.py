from algorithms.users.userFedFTG import UserFedFTG
from algorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, create_generator, aggregate_user_data
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time

MIN_SAMPLES_PER_LABEL = 1

class FedFTG(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)
        self.model_lr = 0.1
        self.gen_lr = 0.01
        self.generative_model = create_generator(dataset=args.dataset, algorithm="FedFTG", embedding=False)
        self.data = read_data(args.dataset)
        self.use_adam = 'adam' in self.algorithm.lower()
        clients = self.data[0]
        self.total_users = len(clients)
        self.total_test_samples = 0
        # train_data, test_data 均为字典类型
        self.train_data = self.data[1]
        self.test_data = self.data[2]
        self.cent_train_dataloader, _, self.available_labels, self.label_counts =\
            aggregate_user_data(self.data, batch_size=self.batch_size)
        # 学习率、权重衰减来自FedFTG中定义
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model_lr, weight_decay=0.001)
        self.generative_optimizer = torch.optim.Adam(self.generative_model.parameters(), lr=self.gen_lr)
        self.lr_scheduler_server = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)
        self.lr_scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(self.generative_optimizer, gamma=0.998)
        self.init_loss_fn()
        # 实例化客户端对象
        for i in range(self.total_users):
            id, train_data, test_data, label_info = read_user_data(i, self.data, count_labels=True)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            user = UserFedFTG(args, id, model,train_data, test_data,
                         self.available_labels, label_info, use_adam=self.use_adam)
            self.users.append(user)
        # 每个客户端的训练集样本数
        self.client_weight = np.array([user.train_samples for user in self.users])
        self.client_weight_array = self.client_weight.reshape((self.total_users, 1))
        self.rand_seed = 1024
        self.init_ensemble_configs(algorithm=args.algorithm)

    def train(self, args):
        for i in range(self.num_glob_iters):
            print("\n\n----------Round number:", i, "----------\n\n")
            inc_seed = 0
            while True:
                # 选择规则不适用于客户端数量较少的情况，生成的随机数并不均匀，致使被选客户端数量不符合应有数量
                np.random.seed(i + self.rand_seed + inc_seed)
                act_list = np.random.uniform(size=self.total_users)
                act_clients = act_list < self.num_users/self.total_users
                # 被选客户端索引
                self.selected_users_idx = np.sort(np.where(act_clients)[0])
                # 被选客户端
                self.selected_users = [self.users[i] for i in self.selected_users_idx]
                inc_seed += 1
                if len(self.selected_users) != 0:
                    break
            print("Selected clients: {}".format(self.selected_users_idx))
            self.send_parameters(mode="all")
            # 训练客户端模型
            for user in self.selected_users:
                # weight_decay 已规定为0.001
                user.train(verbose=True)

            # 聚合被选客户端的参数
            self.aggregate_parameters(partial=False)
            # 训练生成器与服务器
            self.train_generator()
            test_acc, test_loss = self.evaluate(save=True)
            # test_acc_server, test_loss_server = self.evaluate_server(self.test_data, save=True)
            #print("Communication %3d, Test Accuracy: %.4f, Test Loss: %.4f", i, test_acc, test_loss)
           # print("Communication %3d, Server Test Accuracy: %.4f, Server Test Loss: %.4f",
                 # i, test_acc_server, test_loss_server)

            if i > 0 and i % 20 == 0:
                self.visualized_images(self.generative_model, i, repeats=10)
            self.lr_scheduler_server.step()
            self.lr_scheduler_gen.step()

            self.evaluate_server(self.data, save=True)
        self.save_results(args)
        self.save_model()



    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
                label_weights.append(np.array(weights)/np.sum(weights))# 归一化标签计数（权重是不同客户端中同一类的权重）
            else:
                label_weights.append(np.array([0 for _ in range(len(self.selected_users))]))
        #label_weights = np.array(label_weights).reshape(len(qualified_labels), -1)
        label_weights = np.array(label_weights).reshape(self.unique_labels, -1)
        # 标准化计数权重二维列表形状
        # 返回标签权重及合格标签列表（标签权重形状为（类别数，客户端数量））
        # print(label_weights.shape)
        return label_weights, qualified_labels

    def generate_labels(self, number, cls_num):
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)

    def get_batch_weight(self, labels, cls_clnt_weight):
        bs = labels.size
        num_clients = cls_clnt_weight.shape[1]
        batch_weight = np.zeros((bs, num_clients))
        #print(batch_weight.shape, cls_clnt_weight.shape)
        #print(cls_clnt_weight, labels)
        batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
        return batch_weight

    def train_generator(self):
        # student_model= self.model, g_model = self.generative_model, teacher_model为各客户端模型
        batch_size = self.batch_size
        self.model = self.model.cuda()
        for user in self.selected_users:
            user.model.eval()
            user.model = user.model.cuda()
        self.generative_model = self.generative_model.cuda()
        num_clients = self.num_users
        num_classes = self.unique_labels
        # 总的服务器端模型的训练轮数
        iterations = 10
        inner_round_g = 1
        inner_round_d = 5

        for param in self.model.parameters():
            param.require_grads = True
        # cls_clnt_weight形状（类别数，总客户端数量）
        cls_clnt_weight, _ = self.get_label_weights()
        # cls_clnt_weight = cls_clnt_weight.transpose()
        # 迭代所使用的所有标签
        labels_all = self.generate_labels(iterations * self.batch_size, self.label_counts)

        #训练生成器和分类模型
        for e in range(iterations):
            labels = labels_all[e*batch_size:(e*batch_size+batch_size)]
            batch_weight = torch.Tensor(self.get_batch_weight(labels, cls_clnt_weight)).cuda()
            onehot = np.zeros((batch_size, num_classes))
            onehot[np.arange(batch_size), labels] = 1
            y_onehot = torch.Tensor(onehot).cuda()
            # 噪声在生成器模型中
            # 训练生成器
            self.model.eval()
            self.generative_model.train()

            loss_G = 0
            loss_md_total = 0
            loss_cls_total = 0
            loss_ap_total = 0
            y_input = torch.Tensor(labels).long().cuda()
            for _ in range(inner_round_g):
                for client_id, client in enumerate(self.selected_users):
                #for client, client_id in zip(self.selected_users, self.selected_users_idx):
                    self.generative_optimizer.zero_grad()
                    result = self.generative_model(y_input)
                    z = result['eps']
                    fake = result['output']

                    t_logit = client.model(fake)['logit']
                    s_logit = self.model(fake)['logit']
                    # print(s_logit.device, t_logit.detach().device)
                    loss_md = -torch.mean(torch.mean(torch.abs(s_logit-t_logit.detach()), dim=1)
                                          * batch_weight[:, client_id])
                    loss_cls = torch.mean(self.ce_loss(t_logit, y_input) * batch_weight[:, client_id].squeeze())
                    loss_ap = self.generative_model.diversity_loss(z.view(z.shape[0], -1), fake)

                    loss = loss_md + loss_cls + loss_ap / self.num_users
                    loss.backward()

                    loss_G += loss
                    loss_md_total += loss_md
                    loss_cls_total += loss_cls
                    loss_ap_total += loss_ap
                    self.generative_optimizer.step()
            # 训练服务器模型
            self.model.train()
            self.generative_model.eval()
            for _ in range(inner_round_d):
                self.optimizer.zero_grad()
                fake_cls = self.generative_model(y_input)['output'].detach()
                s_logit_cls = self.model(fake_cls)['logit']
                t_logit_merge = 0
                for client_id, client in enumerate(self.selected_users):
                #for client, client_id in zip(self.selected_users, self.selected_users_idx):
                    t_logit_cls = client.model(fake_cls)['logit'].detach()
                    t_logit_merge += F.softmax(t_logit_cls, dim=1) * batch_weight[:, client_id][:, np.newaxis].repeat(1, num_classes)
                loss_D = torch.mean(-F.log_softmax(s_logit_cls, dim=1) * t_logit_merge)
                loss_D.backward()
                self.optimizer.step()

            # 此处测试的是所有客户端的平均精度与损失
            #test_acc, test_loss = self.evaluate(save=True, selected=False)
            # print("Epoch %3d, Test Average Accuracy: %.4f, Loss: %.4f" % (e+1, test_acc, test_loss))
            #self.evaluate_server(self.test_data, save=True)
            # print("Epoch %3d, Test Server Accuracy: %.4f, Loss: %.4f" % (e + 1, test_acc_server, test_loss_server))
            self.model.train()

        for params in self.model.parameters():
            params.requires_grad = True
        self.model.eval()

        return self.model

    def visualized_images(self, generator, glob_iter, repeats=1):
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats, axis=0) # 按列复制数组（扩展行）
        y_input = torch.LongTensor(y)
        generator.eval()
        images = generator(y_input)['output']
        # images.shape[1:]获取images的H、W,*表示拆分成多个元素
        # 转换形状(1，B，H，W)
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, padding=0, normalize=True)
        print(f"Image saved to {path}")