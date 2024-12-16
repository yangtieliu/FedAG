import torch
import torch.nn.functional as F
import numpy as np
import copy
from algorithms.users.userbase import User
from utils.model_utils import create_generator
from tqdm import tqdm


class UserFedAG(User):
    def __init__(self, args, id, model, generative_model, train_data, test_data,
                 available_labels, latent_layer_idx, label_info, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = create_generator(args.dataset, algorithm=args.algorithm, embedding=args.embedding)
        self.generative_model.initialize()
        self.generative_model_optimizer = torch.optim.Adam(self.generative_model.parameters(), lr=0.001)
        self.generative_model_global = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info = label_info
        self.gpu(args)
        self.delta_w = 0
        # 生成器的聚合权重
        self.aggregate_weight = 0.8
        self.layer_idx = 2  # 表示生成器模型共享参数的层数
        self.eta = 0.1  # 用于权重矩阵的更新
        self.thershold = 0.1  # 用于判定权重矩阵是否收敛,设定值需要根据训练轮调整
        self.weights = None

    def gpu(self, args):
        if args.device == 'cuda':
            self.cuda = True
            self.generative_model.cuda()

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        lr = max(init_lr * (decay ** (epoch // lr_decay_epoch)), 0.0001)
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += int(count)

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    def calculate_delta_w(self, model_before, model_after):
        params_before = [param.data for param in model_before.parameters() if param.requires_grad]
        params_after = [param.data for param in model_after.parameters() if param.requires_grad]
        param_before = torch.cat([torch.flatten(p) for p in params_before])
        params_after = torch.cat([torch.flatten(p) for p in params_after])
        # 模型参数之差的二范数中的模型参数究竟该是什么形状暂时不清楚
        norm_delta_w = torch.norm((params_after - param_before), 2)
        self.delta_w = norm_delta_w / torch.norm(param_before, 2).item()
        return self.delta_w

    def train(self, glob_iter, personalized=False, generator_way='weighted', early_stop=100, regularization=True,
              verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.model_before = copy.deepcopy(self.model)
        if generator_way == 'FTSG':  # fine-tuning server generator
            self.generative_model = copy.deepcopy(self.generative_model_global)
        # self.generative_model.eval() 将eval语句转移到epoch轮内
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0

        # 分类模型训练
        for epoch in range(self.local_epochs):
            self.model.train()
            self.generative_model.eval()
            for i in range(
                      int(np.ceil(self.train_samples/self.batch_size))
            ):  # 迭代次数可能需要改为：本地训练集样本数量/batch_size
                self.optimizer.zero_grad()
                samples = self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                self.update_label_counts(samples['labels'], samples['counts'])
                # print()
                model_result = self.model(X)
                user_output_logp = model_result['output']
                # 在本地训练集上的预测损失
                predictive_loss = self.loss(user_output_logp, y)

                # 使用生成器生成样本辅助训练
                if regularization and epoch < early_stop:
                    if glob_iter < 60:
                        generative_alpha = self.exp_lr_scheduler(glob_iter, decay=1.01, init_lr=1)
                        generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.999, init_lr=10)
                    else:
                        generative_alpha = self.exp_lr_scheduler(60, decay=1.01, init_lr=1)
                        generative_beta = self.exp_lr_scheduler(60, decay=0.999, init_lr=10)
                    gen_output = self.generative_model(y ,
                                                       verbose=True)['output']
                    pred_given_gen = self.model(gen_output)['logit']
                    target_p = F.softmax(pred_given_gen, dim=1).clone().detach()
                    # 计算生成器与分类模型在同一标签下上的不同数据的kl损失
                    user_latent_loss = self.ensemble_loss(user_output_logp, target_p)

                    # 此处选择y的目的及分布并不明确，尚待考虑 （随机生成一段指定batch（生成器batch）大小的标签）
                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    if self.cuda:
                        sampled_y = sampled_y.cuda()
                    gen_result = self.generative_model(sampled_y,
                                                       verbose=True)
                    gen_output = gen_result['output']
                    # 在伪数据上的预测向量
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    # 计算伪数据上的预测交叉熵损失函数
                    # 鉴于self.loss()的reduction默认为mean，torch.mean()并无实际意义，不影响输出结果
                    teacher_loss = torch.mean(self.loss(user_output_logp, sampled_y))
                    # 同一样的loss函数，均为负对数似然函数
                    # teacher_loss = torch.mean(self.generative_model.crossentropy_loss(user_output_logp, y))
                    gen_ratio = self.gen_batch_size / self.batch_size
                    # 计算分类器模型的损失
                    loss = predictive_loss + generative_alpha * gen_ratio * teacher_loss + generative_beta * user_latent_loss
                    # 累加TEACHER_LOSS和LATENT_LOSS
                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                else:
                    # 否则不使用伪数据
                    loss = predictive_loss
                # 损失函数反向传播
                loss.backward()
                # 优化器迭代
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()

            # 生成器更新 train generator
            if generator_way == "FTSG":  # fine-tune server_generator
                # 客户端端生成器训练（为服务器发送的生成器的模型）
                self.generative_model.train()
                self.model.eval()
                #self.generative_model = self.generative_model_global
                for i in range(
                        self.K
                ):
                    self.generative_model_optimizer.zero_grad()
                    sample_weights = [list(self.label_counts.keys())[l] for l in self.available_labels]
                    sample_weights = np.array(sample_weights)
                    sample_weights = list(sample_weights / np.sum(sample_weights))
                    #sample_weights = np.sum(sample_weights)/sample_weights  # 样本越少的权重越大 # 未考虑列表元素含0的问题
                    #sample_weights = list(sample_weights/np.sum(sample_weights))  # 归一化
                    y = np.random.choice(self.available_labels, self.gen_batch_size,
                                         p=sample_weights)
                    y = torch.LongTensor(y)
                    if self.cuda:
                        y = y.cuda()
                    #torch.autograd.detect_anomaly(True)
                    #with torch.autograd.detect_anomaly():
                    gen_output = self.generative_model(y, verbose=True)
                    gen_eps = gen_output['eps']
                    #  print(gen_eps)
                    diversity_loss = self.generative_model.diversity_loss(gen_eps.view(gen_eps.shape[0], -1), gen_output['output'])
                    user_output_fake = self.model(gen_output['output'], start_layer_idx=self.latent_layer_idx)
                    gen_loss = self.loss(user_output_fake['output'], y)

                    samples_g = self.get_next_train_batch(count_labels=True)
                    X_g, y_g = samples_g['X'], samples_g['y']
                    if self.cuda:
                        X_g, y_g = X_g.cuda(), y_g.cuda()
                    # self.update_label_counts(samples_g['labels'], samples_g['counts'])
                    # print()
                    model_result_g = self.model(X_g)
                    user_output_logp_g = model_result_g['output']
                    gen_output_g = self.generative_model(y_g, verbose=True)['output']
                    pred_given_gen_g = self.model(gen_output_g)['logit']
                    target_p_g = F.softmax(pred_given_gen_g, dim=1).clone().detach()
                    user_latent_loss_g = self.ensemble_loss(user_output_logp_g, target_p_g)


                    loss_g = - gen_loss - user_latent_loss_g + diversity_loss   # 实验对抗训练
                    loss_g.backward()
                    #torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 10)
                    self.generative_model_optimizer.step()



                #for name, param in self.generative_model.named_parameters():
                    #if torch.isnan(param.grad).any():
                        #print(name, 'grad is nan')
                        # print(torch.max(param))
                    #else:
                        # print('no grad issue')
                        # print(torch.max(param.grad))
            elif generator_way == "weighted":
                for local_param, global_param in zip(self.generative_model.parameters(),
                                                     self.generative_model_global.parameters()):
                    local_param.data = local_param.data + self.aggregate_weight * (global_param.data - local_param.data)
                self.generative_model.train()
                self.model.eval()
                for i in range(self.K):
                    self.generative_model_optimizer.zero_grad()
                    # 按标签占比为采样概率
                    sample_weights = [list(self.label_counts.keys())[l] for l in self.available_labels]
                    sample_weights = np.array(sample_weights)
                    sample_weights = list(sample_weights / np.sum(sample_weights))
                    y = np.random.choice(self.available_labels, self.gen_batch_size,
                                         p=sample_weights)
                    y = torch.LongTensor(y)
                    if self.cuda:
                        y = y.cuda()
                    # 按采样数据生成伪数据并计算损失（伪数据在本地模型上的预测损失）
                    gen_output = self.generative_model(y)
                    user_output_fake = self.model(gen_output['output'], start_layer_idx=self.latent_layer_idx)
                    gen_loss = self.loss(user_output_fake['output'], y)
                    gen_loss.backward()
                    self.generative_model_optimizer.step()
            elif generator_way == "ALA":  # Adaptive Local Aggregation
                self.model.eval()
                self.generative_model.train()
                self.generative_model_global.eval()
                gen_global_param = list(self.generative_model_global.parameters())
                for i in range(self.K):
                    self.generative_model_optimizer.zero_grad()
                    # 按标签占比为采样概率
                    sample_weights = [list(self.label_counts.keys())[l] for l in self.available_labels]
                    sample_weights = np.array(sample_weights)
                    sample_weights = list(sample_weights / np.sum(sample_weights))
                    y = np.random.choice(self.available_labels, self.gen_batch_size,
                                         p=sample_weights)
                    y = torch.LongTensor(y)
                    if self.cuda:
                        y = y.cuda()
                    # 按采样数据生成伪数据并计算损失（伪数据在本地模型上的预测损失）
                    gen_output = self.generative_model(y)
                    user_output_fake = self.model(gen_output['output'], start_layer_idx=self.latent_layer_idx)
                    gen_loss = self.loss(user_output_fake['output'], y)
                    # 测试语句
                    for name, param in self.generative_model.named_parameters():
                        if torch.isnan(param).any():
                            print(name, 'is nan')
                        if param.requires_grad:
                            if torch.isnan(param.grad).any():
                                print(name, 'grad is nan' )
                        else:
                            print(name, "didn't set grad true")
                    print('gen_loss_grad: {}'.format(gen_loss.requires_grad))

                    gen_loss.backward()
                    self.generative_model_optimizer.step()

                gen_local_param = list(self.generative_model.parameters())
                if torch.sum(gen_global_param[0] - gen_local_param[0]) == 0:  # 首轮通信不使用ALA（本地生成器无梯度）
                    print("ALA is not used in the current communication")
                    return

                # 将全局生成器高层参数赋值给本地生成器高层参数
                for local_param, global_param in zip(gen_local_param[-self.layer_idx:],
                                                     gen_global_param[-self.layer_idx:]):
                    local_param.data = global_param.data.clone()
                # 临时生成器模型及对应参数，仅用于训练权重矩阵
                gen_model_temp = copy.deepcopy(self.generative_model)
                params_temp = list(gen_model_temp.parameters())

                # 只对低层参数进行训练
                gen_local_param_p = gen_local_param[:-self.layer_idx]
                gen_global_param_p = gen_global_param[:-self.layer_idx]
                params_temp_p = params_temp[:-self.layer_idx]

                # 冻结高层参数梯度
                for param in params_temp[-self.layer_idx:]:
                    param.requires_grad = False
                # 只计算一次（不需要更新临时模型参数）
                optimizer = torch.optim.SGD(params_temp_p, lr=0)

                # 规定权重矩阵
                if self.weights == None:
                    self.weights = [torch.ones_like(param.data) for param in params_temp_p]

                # 按更新公式更新模型参数
                for param_t, param, param_g, weight in zip(params_temp_p, gen_local_param_p, gen_global_param_p,
                                                           self.weights):
                    param_t.data = param.data + weight * (param_g.data - param.data)

                # 训练权重矩阵
                losses = []
                cnt = 0  # 迭代次数记录
                while True:
                    for i in range(self.K):
                        sample_weights = [list(self.label_counts.keys())[l] for l in self.available_labels]
                        sample_weights = np.array(sample_weights)
                        sample_weights = list(sample_weights / np.sum(sample_weights))
                        y = np.random.choice(self.available_labels, self.gen_batch_size,
                                             p=sample_weights)
                        y = torch.LongTensor(y)
                        if self.cuda:
                            y = y.cuda()
                        optimizer.zero_grad()
                        gen_output = gen_model_temp(y, verbose=True)
                        gen_result = gen_output['output']
                        gen_loss = self.loss(self.model(gen_result)['output'], y)
                        gen_loss.backward()
                        for param_t, param, param_g, weight in zip(params_temp_p, gen_local_param_p, gen_global_param_p,
                                                                   self.weights):
                            #print(param_t.grad[0], param_t.grad.size())
                            weight.data = torch.clamp(
                                (weight - self.eta * param_t.grad * (param_g - param)), 0, 1)
                        # 仅在该batch更新临时模型参数
                        for param_t, param, param_g, weight in zip(params_temp_p, gen_local_param_p, gen_global_param_p,
                                                                   self.weights):
                            param_t.data = param.data + weight * (param_g.data - param.data)
                        losses.append(gen_loss.item())
                    cnt += 1
                    if len(losses) > 100 and np.std(losses[-100:]) < self.thershold:
                        print('Std:', np.std(losses[-100:]), 'ALA epochs:', cnt)
                        break
                    elif cnt > 100:
                        print(np.std(losses[-100:]))
                        print('no convergence for 200 epochs')
                        break
                # 将按权重更新的参数赋值给本地生成器参数
                for param, param_t in zip(gen_local_param_p, params_temp_p):
                    param.data = param_t.data

        self.model_after = copy.deepcopy(self.model)
        # 此处提取的模型参数是否需要使用detach()仍需考虑
        self.delta_w = self.calculate_delta_w(self.model_before, self.model_after)
        if regularization and verbose:
            test_acc, _, num_test_samples = self.test()
            test_acc = test_acc / num_test_samples
            print("clients test_acc:", test_acc)
            # print(self.model)
            # TEACHER_LOSS = TEACHER_LOSS.cpu().detach().numpy()/(self.local_epochs * self.K)
            # LATENT_LOSS = LATENT_LOSS.cpu().detach().numpy()/(self.local_epochs * self.K)
            loss = loss.cpu().detach().numpy()
            info = 'one of clients loss ={:4f}'.format(loss)
            info += '\nUser Teacher Loss ={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            print(info)
