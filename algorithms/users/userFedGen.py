import torch
import torch.nn.functional as F
import numpy as np
from algorithms.users.userbase import User


class UserpFedGen(User):
    def __init__(self,
                 args, id, model, generative_model, train_data, test_data, available_labels,
                 latent_layer_idx, label_info, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)  # 继承userbase的
        self.gen_batch_size = args.gen_batch_size  # 实验中生成器生成batch大小同数据集采样batch大小
        self.generative_model = generative_model  # 生成器模型
        self.latent_layer_idx = latent_layer_idx  # 潜在层索引（即生成倒数第几层的表征）（代码未提供除生成最后一层表征以外的实现代码）
        self.available_labels = available_labels  # 推断为有效标签
        self.label_info = label_info  # 推测为标签信息
        self.gpu(args)

    def gpu(self,args):
        if args.device=='cuda':
            self.cuda=True

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs"""
        # 每lr_decay_epoch轮指数衰减一次学习率，不得低于1e-4
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):  # 更新标签计数，其中labels,counts应为列表或数组
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count  # self.label_counts字典指定键（标签）添加指定数值（计数）

    def clean_up_counts(self):  # 清除标签计数
        del self.label_counts  # 删除self.label_counts的计数
        self.label_counts = {label: 1 for label in range(self.unique_labels)}  # 所有标签对应计数为1

    # 定义用户训练函数，参数全局迭代轮，是否进行个性化训练，早停轮数，正则化，详细信息
    # 该训练函数对应用户损失，如果使用regularization且本地轮小于早停时，损失函数实际为本地样本预测损失+潜在表征预测损失否则为本地样本预测损失
    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True,
              verbose=False):  # glob_iter全局迭代轮
        self.clean_up_counts()  # 清除标签计数
        self.model.train()  # 设定模型为训练模式
        self.generative_model.eval()  # 生成器模型为评估模式
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0  # DIST_LOSS不明确统计的是何种损失(DIST_LOSS用于计算KL散度，LATENT_LOSS用于计算生成器损失)
        for epoch in range(self.local_epochs):  # 在本地训练轮数期间
            self.model.train()  # 模型训练模式
            for i in range(
                    1#int(np.ceil(self.train_samples/self.batch_size))#self.K
            ):  # batch个数？可能为本地迭代次数。根据main.py给出的解释，为'computation steps'
                self.optimizer.zero_grad()  # 指定优化器梯度为0（清空梯度）
                ###sample from real dataset (un-weighted)
                # 从0开始循环采样batch，返回的是字典，形式为：{'X': X, 'y': y}，其中X,y为iter(self.trainloader)中的一组batch
                # count_labels为真表示在字典中添加'labels','counts'键，统计标签类别及每个类别对应的数量
                samples = self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']  # 提取该batch中的样本及标签
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                self.update_label_counts(samples['labels'], samples['counts'])  # 根据标签列表和对应计数列表更新标签计数
                #if self.model_name == 'resnet18':
                    #model_result = {}
                    #model_result['logit'] = self.model(X)
                    #model_result['output'] = F.log_softmax(model_result['logit'], dim=1)
                #else:
                model_result = self.model(X, logit=True)  # 输出模型输出及logit向量
                user_output_logp = model_result['output']  # 输出模型经log_softmax的最终输出
                predictive_loss = self.loss(user_output_logp, y)  # 计算预测损失

                ####sample y and generate z 采样y并生成z
                if regularization and epoch < early_stop:  # early_stop作用不明确
                    # alpha和beta超参数作为学习率进行逐轮指数衰减
                    generaltive_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, #init_lr=1)
                                                              init_lr=self.generative_alpha)
                    generaltive_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, #init_lr=10)
                                                             init_lr=self.generative_beta)
                    ###get generaltive_model(latent_representation) of the same label#获取相同标签的生成器模型（潜在表征）
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)[
                        'output']  # 返回生成器模型生成的指定层的潜在表征
                    #logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, flag=True,
                                                # logit=True)['logit'] ### resnet18使用
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx,
                                                 logit=True)['logit']  # 按该潜在表征所在层数决定其输入至用户模型的第几层以获得模型输出（logit)
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()  # 对输出进行softmax，并赋值一份以从计算图中脱离（移除梯度计算）
                    user_latent_loss = generaltive_beta * self.ensemble_loss(user_output_logp,
                                                                             target_p)  # 计算用户在真实样本上的预测值和通过该样本生成的潜在表征在模型上的输出之间的KL散度，并乘以β

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)  # 从所有可选标签中选择生成样本batch大小的数据
                    sampled_y = torch.tensor(sampled_y)  # 转换为张量（形式由数组元素类型决定）（可能为整型）
                    if self.cuda:
                        sampled_y=sampled_y.cuda()
                    gen_result = self.generative_model(sampled_y,
                                                       latent_layer_idx=self.latent_layer_idx)  # 使用采样的标签生成潜在表征，是一组batch的潜在表征
                    gen_output = gen_result[
                        'output']  # latent representation when latent = True, x otherwise（此处备注不对，参见generatora前向传播备注）获得潜在表征
                    #user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx, flag=True)[
                      #  'output']
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)[
                        'output']  # 这里的user_output_logp是生成器生成的潜在表征为输入的模型输出
                    teacher_loss = generaltive_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp,
                                                                sampled_y))  # 此处交叉熵损失函数为nn.NLLLoss()，取均值再乘以alpha
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size  # 计算生成样本和真实样本batch的比值
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss  # teacher_loss对应在生成器batch大小上损失，user_latent_loss在真实样本batch大小上损失
                    TEACHER_LOSS += teacher_loss  # 累加教师损失
                    LATENT_LOSS += user_latent_loss  # 累加用户在样本上预测值和在潜在表征上预测值的KL散度为损失的值
                else:
                    ###get loss and perform optimization#获取损失并执行优化
                    loss = predictive_loss  # 以在真实样本上的预测损失为训练损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # self.local_model，更新优化器
        # local_model<===self.model 从完成训练的模型中提取参数
        self.clone_model_parameters(self.model.parameters(), self.local_model)  # 逐层复制训练模型参数至self.local_model
        self.lr_scheduler.step(glob_iter)  # 更新学习率（当前的全局轮数）
        if regularization and verbose:  # 如果正则化且有详细信息
            TEACHER_LOSS = TEACHER_LOSS.cpu().detach().numpy() / (self.local_epochs * self.K)  # 本地训练在每个本地轮和每个客户端上的教师损失
            LATENT_LOSS = LATENT_LOSS.cpu().detach().numpy() / (self.local_epochs * self.K)  # 本地训练在每个本地轮和每个客户端上的潜在损失
            info = '\nUser Teacher Loss= {:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            print(info)  # 输出上述两损失

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples[
            'counts']  # 一个batch长度内不同的标签及对应计数，来自samples=self.get_next_train_batch()
        # weight=self.label_weights[y][:,user_idx].reshape(-1,1)
        np_y = samples['y'].detach().numpy()  # 转换标签列表为数组
        n_labels = samples['y'].shape[0]  # 标签列表长度，即batch长度
        weights = np.array([n_labels / count for count in counts])  # smaller count --> larger weight，越少的样本权重越大
        weights = len(self.available_labels) * weights / np.sum(weights)  # normalized 归一化后乘可获得标签列表长度
        label_weights = np.ones(self.unique_labels)  # 创建标签权重列表全为1，长度为独特标签的总数（来自预先配置）
        # labels为数组（为按从小到大排序后的标签类别数组），weights也为数组，将标签列表索引对应标签的权重置为乘以标签列表长度的权重列表
        # 可用数组/列表/元组作为索引获取数组对应值，返回的还是数组。（一维数组）labels元素范围不能超过label_weights的长度，weights与labels长度一致。
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]  # 根据标签权重对应的标签列表索引，获取标签列表索引对应标签的权重
        return sample_weights