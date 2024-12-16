from algorithms.users.userFedGen import UserpFedGen
from algorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time

MIN_SAMPLES_PER_LABEL = 1


class FedGen(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)  # 继承Server父类的初始函数

        # Initialize data for all users
        data = read_data(args.dataset)  # 读取的数据集为划分后的数据集
        self.data = data
        # data contains:clients, groups, train_data, test_data, proxy_data
        # data包含：客户端（列表），组（列表），训练数据集，测试数据集，代理数据集（字典）
        clients = data[0]
        total_users = len(clients)  # 总客户端数量，已由划分数据集时决定
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()  # 为真时赋值成立
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs 早停轮用于生成样本的指定训练轮数
        # 学生模型参数深拷贝当前训练模型。self.model来自Server定义，具体使用来自model_utils中的create_model，包括创建的模型及模型名称
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generator(args.dataset, args.algorithm, args.embedding)
        if args.device == 'cuda':
            self.generative_model = self.generative_model.cuda()
            self.cuda = True
        if not args.train:  # 如果不处于训练模式下，返回生成器和服务器训练模型的参数
            print('number of generator parameters: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameters: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx  # 指定生成倒数第几层的潜在表征
        self.init_ensemble_configs("FedGen")  # 初始化服务器配置（即从model_config读取参数）
        print("latent_layer_idx:{}".format(self.latent_layer_idx))  # 打印指定倒数第几层
        print("label embedding {}".format(self.generative_model.embedding))  # 是否使用生成器标签嵌入（即将单值标签映射为张量，参见generator.py）
        print("ensemeble learning rate: {}".format(self.ensemble_lr))  # 打印（生成器？）学习率及服务器端alpha,beta,eta
        print("ensemeble alpha={},beta={},eta={}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generate alpha={}, beta={}".format(self.generative_alpha, self.generative_beta))
        # 生成器alpha,beta（用于计算“客户端”损失函数），参见userpFedGen
        self.init_loss_fn()  # 初始化损失函数
        # 返回训练数据dataloader及迭代器和对应不同标签组成的列表，该函数与read_data()相关
        self.train_data_loader, self.train_iter, self.available_labels, _ = aggregate_user_data(data,
                                                                                                self.ensemble_batch_size)
        # 生成器优化器选择adam，weight_decay初始为0
        # lr=self.ensemble_lr为初始学习率，设置为1e-4，同时也为常用adam优化器的初始学习率
        self.generative_optimizer = torch.optim.Adam(params=self.generative_model.parameters(),
                                                     lr=self.ensemble_lr, betas=(0.9, 0.999),
                                                     eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        # 生成器学习率衰减选择指数衰减，指数衰减指数为0.98
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98
        )
        # 服务器模型训练优化器选择Adam（未使用）
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False
        )
        # 模型训练学习率衰减选择指数衰减，指数衰减指数为0.98（未使用）
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        # creating users ###
        # 创建客户端列表
        self.users = []
        for i in range(total_users):  # 对于每个客户端
            # 读取指定客户端数据，返回客户端名称，训练集，测试集，标签信息
            id, train_data, test_data, label_info = read_user_data(i, data, count_labels=True)
            self.total_train_samples += len(train_data)  # 累加总客户端训练样本数量
            self.total_test_samples += len(test_data)  # 累加总客户端测试样本数量
            id, train, test = read_user_data(i, data, count_labels=False)  # ？？？？此处是否多余
            # 创建客户端类
            user = UserpFedGen(
                args, id, model, self.generative_model, train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam)  # 使用adam优化器/

            self.users.append(user)  # 添加至客户端列表
        # 打印总客户端训练/测试样本数量
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        # 打印总客户端数量
        print("Data from {} users in total.".format(total_users))
        # 打印结束创建FedGen的服务器
        print("Finished creating FedGen Server.")

    # 服务器训练函数
    def train(self, args):
        ###pretraining 预训练
        for glob_iter in range(self.num_glob_iters):  # 在全局通信轮中
            print("\n\n----------Round number:", glob_iter, "------------\n\n")  # 打印当前通信轮
            # self.num_users=args.num_users，来自Server，为外部定义。返回被选客户端对象列表，对应客户端索引（元素为整数，非f_0000x）
            # 该函数将self.num_users与len(self.users)比较大小,self.users长度由之前划分的数据集决定，self.users此处已继承至fedgen。
            # 参见model_utils.py
            self.selected_users, self.user_idxs = self.select_users(self.num_users, way='random', return_idx=True)
            # 如果self.local不为真，则按是否进行完整参数传递（'all','decode'）决定是否进行完整模型参数传递
            # 传递方向为从服务器的self.model到客户端的self.model
            if not self.local:
                self.send_parameters(mode=self.mode)  # broadcast averaged prediction model
            # 执行Server的evaluate()函数，打印并保存当前客户端全局平均准确率和损失值
            self.evaluate()
            # 在所有用户列表长度大小内随机选择一个整数 #可考虑将len(self.users)改为self.user_idxs
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time()  # log user-training start time 返回所有客户端开始训练当前时间（以秒为单位）
            # user_id为整数，user为对应的客户端对象。对被选客户端进行训练

            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                # verbose当user_id==chosen_verbose_user时为真，verbose=True表示打印训练信息(Teacher loss,latent loss)
                verbose = user_id == chosen_verbose_user
                # 在第一轮通信后，使用生成器样本进行正则化
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,  # 全局轮
                    personalized=False,  # self.personalized来自Server，为真时将模型参数复制到个性化模型中
                    early_stop=self.early_stop,  # 20，local_epochs实际训练中也为20
                    verbose=verbose,  # 为真时输出参数
                    regularization=glob_iter > 0  # 除首轮外，为真
                )
            curr_timestamp = time.time()  # log user_training end time 所有客户端训练结束时刻
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)  # 每个客户端的平均训练时长
            self.metrics['user_train_time'].append(train_time)  # metrics添加客户端平均训练时间
            # 评估个性化模型
            # if self.personalized:
            # self.evaluate_personalized_model()

            self.timestamp = time.time()  # 生成器模型训练及模型聚合开始时间
            self.train_generator(
                self.batch_size,  # self.batch_size=args.batch_size，为终端输入参数，根据实际实验，此处batch等于客户端batch
                epoches=self.ensemble_epochs // self.n_teacher_iters,  # 根据配置文件：epoches=50//5
                latent_layer_idx=self.latent_layer_idx,  # 倒数第几层的潜在表征和从倒数第几层输入到客户端模型
                verbose=True  # 打印教师损失，学生损失，多样化损失
            )
            self.aggregate_parameters(
                partial=False if self.mode == 'all' else True)  # 将服务器模型参数替换为所有客户端模型（或共享层'decode_fc2'）的加权和
            curr_timestamp = time.time()  # log server-agg end time# 生成器模型训练及模型聚合结束时间
            agg_time = curr_timestamp - self.timestamp  # 聚合（和生成器训练）所用时间
            self.metrics['server_agg_time'].append(agg_time)  # 添加服务器聚合时间至self.metrics字典
            # 如果全局轮>0且全局轮为20的倍数且self.latent_layer_idx（决定生成器生成的表征是倒数第几层的）为0（生成原始图片）时
            if glob_iter > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                # 使用可视化图片
                self.visualized_images(self.generative_model, glob_iter, repeats=10)
            self.evaluate_server(self.data, True)

        # 保存参数信息和模型信息
        self.save_results(args)  # 保存参数及结果信息至results文件夹以h5文件保存
        self.save_model()  # 保存模型信息至model文件夹并创建数据集信息文件夹下的server.pt

    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        # 生成器训练函数
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'
        :param batch_size:
        :param epoches:
        :param latent_layer_idx:if set to -1(-2),get latent representation of the last(or 2nd to last)layer
        :param verbose:print loss information
        :return: Do not return anything
        """
        # self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()  # 获取每类标签在客户端间的权重二维数组和合格标签列表
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0  # 教师损失，学生损失，多样化损失，第二学生损失

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()  # 生成器模型设置为训练模式
            student_model.eval()  # 学生模型设置为测试模式
            for i in range(n_iters):  # n_iters使用中为self.n_teacher_iters，在serverbase.py中定义为5
                self.generative_optimizer.zero_grad()  # 清零梯度
                y = np.random.choice(self.qualified_labels,
                                     batch_size)  # [ , ,..., ]长为batch_size的数组,重复选取，self.qualified_labels为列表
                y_input = torch.LongTensor(y)  # 改为长整型张量，长度为batch_size
                if self.cuda:
                    y_input = y_input.cuda()
                #print("y_input size", y_input.shape)
                ##feed to generator,可能是generator的forward（）对应部分
                # latent_layer_idx在generator.py中属于未使用参数，返回结果为潜在表征（此处即为最后一层）
                gen_result = self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation Z(latent) if latent set to True， X(raw image) otherwise（备注问题，是否获得表征由latent_layer_idx决定
                gen_output, eps = gen_result['output'], gen_result['eps']  # 'eps'由verbose参数决定是否记录，默认为真，记录噪声信息
                # print(gen_output.shape)
                ###get losses ###
                # decoded=self.generative_regularizer(gen_output)
                # regularization_loss=beta*self.generative_model.dist_loss(decoder,eps)
                # diversity_loss对应generator.py中的DiversityLoss()
                # eps和gen_output的损失值为eps张量行之间的l2距离*生成器输出张量行之间的l1距离的平均值取指数
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                #######get teacher loss#######
                teacher_loss = 0
                teacher_logit = 0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    # y为一维数组，self.label_weights[y]即按y中的元素值对self.label_weights中对应一行进行索引
                    # （代表该类数据在所有被选客户端中的权重）返回二维数组。再使用[:,user_idx]索引该客户端对应y中类别的权重
                    # 并使用reshape(-1,1)从一维数组改变为二维数组
                    weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                    # np.tile(A,reps)扩展目标数组A,reps指重复次数，整数即横向扩展，二维元组则按（纵向平铺，横向平铺）规则重复
                    # 参考:http://t.csdnimg.cn/eaAq3
                    # 此处将该客户端的权重按行，横向复制self.unique_labels次，形状为(batch_size,self.unique_labels)类似（B,CLASSES)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    # 返回该客户端模型在生成器模型输出为输入的情况下的输出结果，此出logit=True返回字典包括'logit'和'output'
                    #if self.model_name == 'resnet18':
                        #user_result_given_gen = {}
                        #user_result_given_gen['logit'] = user.model(gen_output)
                        #user_result_given_gen['output'] = F.log_softmax(user_result_given_gen['logit'])
                    #else:
                    ###user_result_given_gen = user.model(gen_output, flag=True, start_layer_idx=latent_layer_idx, logit=True)
                    user_result_given_gen = user.model(gen_output, start_layer_idx=latent_layer_idx,
                                                       logit=True)
                    # 对logit张量按行进行对数softmax(logit实际上就是最后一层的表征)
                    #print(user_result_given_gen.shape)
                    user_output_logp_ = F.log_softmax(user_result_given_gen['logit'], dim=1)
                    # 计算该客户端的教师损失，客户端在生成样本上的输出结果（B,Classes）与训练该生成样本的y_input(B,1)之间的交叉熵损失(NLL_LOSS)
                    # 因为模型输出经过了log_softmax，所以采用NLLLoss，
                    # 且由于该损失函数定义中reduce=False，所以交叉熵返回形状为张量(B,),否则直接返回所有样本损失平均值。
                    # 再乘以各类样本权重(一一对应y_input)获得平均值作为教师损失值
                    # 乘以权重的目的用于强化该类样本在所有客户端中权重较大的客户端的训练作用
                    teacher_loss_ = torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) *
                        torch.tensor(weight, dtype=torch.float32).cuda()
                    )
                    # 累加各客户端的教师损失值
                    teacher_loss += teacher_loss_
                    # 累加各客户端的经加权的输出logit(B,Classes),expand_weight为(B,Classes),对张量*为元素相乘，size不一致触发广播机制
                    teacher_logit += user_result_given_gen['logit'] * torch.tensor(expand_weight,
                                                                                   dtype=torch.float32).cuda()

                #####get student loss ####获取学生损失
                # 获取学生模型在生成样本上的输出（学生模型是啥？），根据本函数的使用，应为服务器模型
                ### resnet18模型时启用
                # student_output = student_model(gen_output, start_layer_idx=latent_layer_idx, flag=True, logit=True)
                student_output = student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                # 将学生模型（服务器模型）的log_softmax输出与教师模型（客户端模型加权和）的softmax输出进行KL散度损失计算
                # kl_div()中input和targets张量形状相同，input做softmax操作后还需要做log操作.
                # 参数reduction默认为'mean'，返回除以输出张量元素个数的平均值，‘batchmean'返回除以batch_size大小的平均值，'sum'返回张量之和
                # teacher_logit是否应当使用log_softmax以归一化
                student_loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1), F.softmax(teacher_logit, dim=1))
                if self.ensemble_beta > 0:  # ensemble_beta与student_loss相关，决定是否使用student_loss
                    loss = self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()  # 反向传播
                self.generative_optimizer.step()  # 生成器优化器更新
                # 累加加权的教师，学生，多样化损失值
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss  # (torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss  # (torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss  # (torch.mean(diversity_loss.double())).item()
            # 返回加权后的教师，学生，多样化损失值
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):  # 在生成器训练轮中，更新教师损失，学生损失，多样化损失
            # self.n_teacher_iters=5,self.model为服务器模型，TEACHER_LOSS,STUDENT_LOSS,DIVERSITY_LOSS均初始化为0
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        # self.n_teacher_iters*epoches即TEACHER_LOSS,STUDENT_LOSS,DIVERSITY_LOSS损失值累加的次数，这里除以即求平均值
        TEACHER_LOSS = TEACHER_LOSS.cpu().detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.cpu().detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.cpu().detach().numpy() / (self.n_teacher_iters * epoches)
        # 打印生成器：教师损失，学生损失，多样化损失
        info = ("Generator: Teacher Loss={:.4f}, Student Loss:{:.4f}, Diversity Loss:{:.4f},".
                format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS))
        # verbose为真时打印
        if verbose:
            print(info)
        # 生成器学习率优化器更新
        self.generative_lr_scheduler.step()

    def get_label_weights(self):  # 获取标签权重
        label_weights = []  # 标签权重列表
        qualified_labels = []  # 合格标签列表
        for label in range(self.unique_labels):  # 对于所有独特标签循环
            weights = []  # 一类标签在客户端中的权重列表
            for user in self.selected_users:  # 对于所有被选客户端列表
                weights.append(user.label_counts[label])  # 添加每个客户端该标签的计数
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:  # 如果该标签所有客户端中最多样本大于最小样本数阈值
                qualified_labels.append(label)  # 则将该标签添加到合格标签列表
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))  # 将归一化的该标签权重添加至标签权重列表
        label_weights = np.array(label_weights).reshape(
            (self.unique_labels, -1))  # 将所有标签权重列表reshape成(self.unique_labels,被选客户端列表长度)数组
        return label_weights, qualified_labels  # 返回标签权重及合格标签列表

    def visualized_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualized data for a generator
        """
        os.system("mkdir -p images")  # 在命令行执行其中语句
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        # np.repeat(a,repeats,axis=None),重复数组中的元素，axis未指定情况下，默认重复数组中的元素repeats次，组成行向量。
        # 当指定axis时，repeats可为整数或整数组成的数组，axis=1按行进行复制元素，axis=0按列。repeats为整数时，即表示每个元素复制次数；
        # 为数组时，在指定方向，对不同元素复制不同次数 参考：http://t.csdnimg.cn/cvDa9   http://t.csdnimg.cn/UCGq8
        y_input = torch.tensor(y)
        generator.eval()
        images = generator(y_input, latent=False)['output']  # 0,1,...,K 0,1,...,K]#latent参数不清楚来自于哪个函数
        images = images.view(repeats, -1, *images.shape[1:])  # images.shape[1:]获取images的H、W，*此处应指将H、W组成元组
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        # 该句前两个参数为图片和路径，后两个参数为make_grid()的参数，用于生成雪碧图（sprite image），即一组小图片组成的大图。
        # nrow表示大图片中每行所包含的小图片的个数，normalize表示将图片转换至（0，1）之间
        print("Image saved to {}".format(path))
