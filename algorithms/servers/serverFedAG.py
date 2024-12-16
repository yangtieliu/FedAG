from algorithms.users.userFedAG import UserFedAG
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
# np.random.seed(114514)


class FedAG(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)
        self.algorithm = args.algorithm
        self.data = read_data(args.dataset)
        # data : clients, train_data, test_data
        clients = self.data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()  # 为FedGen的模型参数共享规则的标识
        self.use_adam = 'adam' in self.algorithm.lower()
        self.early_stop = 20 # 本地训练经过早停轮后不再使用生成样本(来自FedGen)
        self.student_model = copy.deepcopy(self.model)  # 此处指服务器模型
        if self.algorithm == "FedGen":
            self.latent_layer_idx = -1
        else:
            self.latent_layer_idx = 0
        self.generative_model = create_generator(args.dataset, algorithm=args.algorithm, embedding=args.embedding)
        # self.generative_model.initialize()
        if args.device == 'cuda':
            self.generative_model = self.generative_model.cuda()
            self.cuda = True
        if not args.train:
            print('number of generator parameters: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameters: [{}]'.format(self.student_model.get_number_of_parameters()))
        self.init_ensemble_configs(algorithm=args.algorithm)
        print("label embedding {}".format(self.generative_model.embedding))
        print("generator learning rate: {}".format(self.generator_lr))
        print("generator iteration times: {}".format(self.generator_K))
        print("generator batch size: {}".format(self.generator_batch_size))
        self.init_loss_fn()
        # 客户端训练数据的集合的对应训练迭代器及标签类别统计及对应各标签计数
        self.train_loader, self.train_iter, self.available_labels, self.label_counts = aggregate_user_data(self.data, self.batch_size)
        print(self.label_counts)
        self.generative_optimizer = torch.optim.Adam(params=self.generative_model.parameters(), lr=self.generator_lr,
                                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay,
                                                     amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.generative_optimizer,
                                                                              gamma=0.98)
        # 来自FedFTG优化器设置
        #self.student_optimizer = torch.optim.SGD(params=self.student_model.parameters(), lr=0.01, weight_decay=0)
        self.student_optimizer = torch.optim.Adam(params=self.student_model.parameters(), lr=self.learning_rate,
                                                  betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay,
                                                  amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.student_optimizer, gamma=0.98)
        self.users = []
        self.user_idxs = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, self.data, count_labels=True)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            user = UserFedAG(args, id, model, self.generative_model, train_data, test_data,
                         self.available_labels, self.latent_layer_idx, label_info, use_adam=self.use_adam)
            self.users.append(user)
        print("number of train/test samples:", self.total_train_samples, self.total_test_samples)
        print("total number of users:", len(self.users))
        print("finished creating Server")



    def train(self, args):
        self.selected_users_idx = []
        for glob_iter in range(self.num_glob_iters):
            print("\n\n----------Round number:", glob_iter, "----------\n\n")

            # 根据全局通信轮决定选择哪些客户端
            if glob_iter == 0:
                self.selected_users = self.users
                selected_user_priority = [1 for _ in range(len(self.users))]
            else:
                if glob_iter <= 200:
                    self.selected_users = np.random.choice(self.users, self.num_users, replace=False)
                    #self.selected_users_idx = np.where(np.isin(self.users, self.selected_users))[0]
                    selected_user_priority = [1 for _ in range(self.num_users)]
                else:
                    self.priority = []
                    self.user_idxs = []
                    # self.user_idxs 和 self.priority需每轮清空，避免每次计算优先级会累加
                    # 获取每个客户端的优先级
                    self.get_client_priority(self.users)
                    # 根据优先级选择指定数量客户端
                    self.selected_users, self.user_idxs = self.select_users(self.num_users, self.mode, return_idx=True)
                    # 返回优先级最高的前指定数量个的客户端对应的优先级值
                    selected_user_priority = self.priority

            # 发送全局模型
            self.model = copy.deepcopy(self.student_model)
            self.send_parameters(mode='all')
            self.evaluate(selected=False)
            chosen_verbose_user = np.random.choice(self.selected_users)
            start_time = time.time()  # 客户端训练开始时刻

            # train users
            for user_id, user in enumerate(self.users):
                verbose = user_id == chosen_verbose_user
                user.train(glob_iter,
                           generator_way=self.mode if not glob_iter == 0 else "FTSG",  # 表示生成器训练方式
                           verbose=True,
                           early_stop=self.early_stop,
                           regularization=glob_iter > 0)  # 不激活客户端使用伪数据训练

                # 客户端模型nan检测
                for name, param in user.model.named_parameters():
                    if torch.isnan(param).any():
                        print("user {} model param has nan".format(user_id), name)
                    if torch.isnan(param.grad).any():
                        print("user {} model param.grad has nan".format(user_id), name)

            current_time = time.time()
            train_time = (current_time-start_time)/len(self.users)
            self.metrics['user_train_time'].append(train_time)

            # train generator
            gen_start_time = time.time()
            print()
            self.train_generator(self.batch_size,   # 不采用self.generator_batch_size
                                 selected_user_priority,
                                 way='weighted_labels',  #'priority'
                                 epoches=self.generator_epoches,
                                 verbose=True)
            self.aggregate_parameters(partial=False)  # 此处聚合被选客户端的参数
            self.student_model = copy.deepcopy(self.model)
            gen_end_time = time.time()
            gen_training_time = gen_end_time - gen_start_time
            self.metrics['server_agg_time'] = gen_training_time
            self.generative_model.eval()  # 自行补充"""
            # self.aggregate_parameters(partial=False)

            # train server model(student model)
            server_training_start_time = time.time()
            cls_clnt_weight, qualified_labels = self.get_label_weights()
            self.total_label_weights = self.label_counts/np.sum(self.label_counts)
            for epoch in range(self.generator_epoches):
                self.student_model.train()
                for i in range(self.server_K):
                    total_train = 0
                    for user in self.selected_users:
                        total_train += user.train_samples
                    self.student_optimizer.zero_grad()
                    qualified_label_weights = [self.total_label_weights[i] for i in qualified_labels]
                    qualified_label_weights = np.array(qualified_label_weights)
                    qualified_label_weights = qualified_label_weights / np.sum(qualified_label_weights)
                    y_input = np.random.choice(qualified_labels, size=self.batch_size,
                                               p=list(qualified_label_weights))
                    #batch_weight = torch.Tensor(self.get_batch_weight(y_input, cls_clnt_weight)).cuda()
                    y_input = torch.LongTensor(y_input)
                    gen_result = self.generative_model(y_input, verbose=True)
                    gen_output = gen_result['output']
                    student_output = self.student_model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                    #prediction_loss = self.ce_loss(student_output['output'], y_input.cuda())   # 补充预测损失
                    teacher_logit = torch.zeros((self.batch_size, self.unique_labels)).cuda()
                    for user in self.selected_users:
                        users_output = user.model(gen_output,
                                                  start_layer_idx=self.latent_layer_idx, logit=True)
                        ### 此处蒸馏权重影响甚大，实验中并未使用权重，即均乘1
                        teacher_logit += (users_output['logit'] *
                                          #user.train_samples / total_train * len(self.selected_users))
                                          user.priority)
                    student_loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1), teacher_logit)
                    loss = student_loss #+ prediction_loss
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.student_optimizer.step()

            # 服务器模型nan检测
            for name, param in self.student_model.named_parameters():
                if torch.isnan(param).any():
                    print('server model {} param has nan'.format(name))
                if glob_iter > 0 :
                    if torch.isnan(param.grad).any():
                        print('server model {} param grad has nan'.format(name))

            server_training_end_time = time.time()
            server_training_time = server_training_end_time - server_training_start_time
            self.metrics['server_training_time'] = server_training_time
            self.student_model.eval()
            # 测试服务器模型精度
            self.model = copy.deepcopy(self.student_model)
            self.evaluate_server(self.data, save=True)

            if glob_iter > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualized_images(self.generative_model, glob_iter, repeats=10)
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
        label_weights = np.array(label_weights).reshape(self.unique_labels, -1)
        # 标准化计数权重二维列表形状
        # 返回标签权重及合格标签列表
        return label_weights, qualified_labels

    def get_batch_weight(self, labels, cls_clnt_weight):
        bs = labels.size
        num_clients = cls_clnt_weight.shape[1]
        batch_weight = np.zeros((bs, num_clients))
        #print(batch_weight.shape, cls_clnt_weight.shape)
        #print(cls_clnt_weight, labels)
        batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
        return batch_weight


    def train_generator(self, batch_size, users_priority, way, epoches=1, verbose=False):
        if way == 'weighted_labels':
            self.label_weights, self.qualified_labels = self.get_label_weights()
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = 0, 0, 0
        elif way == 'priority':
            _, self.qualified_labels = self.get_label_weights()
            self.weights = users_priority # 客户端优先级
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = 0, 0, 0
        # 定义函数便于外部输入生成器训练参数
        def update_generator(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            # 所有客户端的标签计数概率分布
            self.total_label_weights = self.label_counts/np.sum(self.label_counts)
            # print(self.label_counts)

            for i in range(n_iters):
                self.generative_optimizer.zero_grad()  # 清空生成器优化器梯度
                if way == 'weighted_labels':
                    # FedGen随机选择一组有效标签数组中的标签，服从标准正态分布选择
                    # 根据FedFTG，进行类似操作：按所有客户端中不同类别样本的概率进行随机生成

                    qualified_label_weights = [self.total_label_weights[i] for i in self.qualified_labels]
                    qualified_label_weights = np.array(qualified_label_weights)
                    qualified_label_weights = qualified_label_weights / np.sum(qualified_label_weights)
                    y = np.random.choice(self.qualified_labels, size=self.batch_size,
                                         p=list(qualified_label_weights))
                    y_input = torch.LongTensor(y)
                    # 此处随机选择的权重需要调整，可能出现权重列表之和不为1的问题
                    #y = np.random.choice(self.qualified_labels, batch_size,
                                         #p=[self.total_label_weights[i] for i in self.qualified_labels])
                    #y_input = torch.LongTensor(y)
                    if self.cuda:
                        y_input = y_input.cuda()
                    # 获取生成器生成的伪样本
                    gen_result = self.generative_model(y_input)
                    gen_output, eps = gen_result['output'], gen_result['eps']
                    # 为了统一方便计算多样化损失而改变了噪声的形状
                    # print(gen_output.size(), eps.size())
                    # 来自FedFTG的噪音使用方法
                    diversity_loss = self.generative_model.diversity_loss(eps.view(eps.shape[0], -1), gen_output)

                    teacher_loss = 0
                    teacher_logit = 0
                    total_train = 0

                    for user in self.selected_users:   # 测试使用样本数作为权重的情况
                         total_train += user.train_samples

                    for user_idx, user in enumerate(self.selected_users):
                        user.model.eval()

                        # 按数组y(长度为batch_size)中的值索引self.label_weights中对应客户端不同类别的权重
                        weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                        # 横向复制权重数组为(batch_size, self.unique_labels) 用于对未经softmax化的输出加权计算
                        expand_weight = np.tile(weight, (1, self.unique_labels))
                        user_result_give_gen = user.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                        user_output_logp = user_result_give_gen['output']
                        # 此处crossentropy_loss()为负对数似然损失函数

                        #weight = user.train_samples / total_train  # 权重改为样本数为权重
                        #expand_weight = weight

                        teacher_loss_ = torch.mean(self.generative_model.crossentropy_loss(user_output_logp, y_input) *
                                                   torch.tensor(weight, dtype=torch.float32).cuda())
                        teacher_loss += teacher_loss_.cuda()
                        teacher_logit += user_result_give_gen['logit'] * torch.tensor(expand_weight,
                                                                                      dtype=torch.float32).cuda()
                    student_output = student_model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                        # 此处的KL散度中必须选择logsoftmax对输出操作，以归一化得概率分布
                        # 如果使用nn.KLDivLoss()则不需要对输出进行softmax
                    student_loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1),
                                            F.softmax(teacher_logit, dim=1))  # student_loss此处为softmax不为nan，log_softmax变为nan

                    loss = (self.ensemble_alpha * teacher_loss + self.ensemble_beta * student_loss
                            + self.ensemble_eta * diversity_loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 10)
                    self.generative_optimizer.step()
                    TEACHER_LOSS += self.ensemble_alpha * teacher_loss
                    STUDENT_LOSS += self.ensemble_beta * student_loss
                    DIVERSITY_LOSS += self.ensemble_eta * diversity_loss
                elif way == 'priority':
                    # torch.autograd.set_detect_anomaly(True)  # 异常值检测
                    sample_weights = [list(self.label_counts)[l] for l in self.qualified_labels]
                    sample_weights = np.array(sample_weights)
                    sample_weights = list(sample_weights / np.sum(sample_weights))
                    y = np.random.choice(self.qualified_labels, batch_size,
                                         p=sample_weights)
                    y_input = torch.LongTensor(y)
                    # print("now in iteration", i, self.label_counts)
                    # print(y_input)
                    if self.cuda:
                        y_input = y_input.cuda()
                    gen_result = self.generative_model(y_input)#, algorithm=self.algorithm
                    gen_output, eps = gen_result['output'], gen_result['eps']
                    # print(gen_output, gen_output.size())
                    diversity_loss = self.generative_model.diversity_loss(eps.view(eps.shape[0], -1), gen_output)
                    teacher_loss = 0
                    teacher_logit = torch.FloatTensor(self.batch_size, self.unique_labels).zero_().cuda()
                    for user_idx, user in enumerate(self.selected_users):
                        user.model.eval()
                        # 归一化优先级值
                        # print('self.weights', self.weights, len(self.weights))
                        weight = self.weights[user_idx]/np.sum(np.array(self.weights))
                        expand_weight = np.tile(weight, (1, self.unique_labels))
                        #print(expand_weight, expand_weight.shape)
                        user_result_give_gen = user.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                        user_output_logp = user_result_give_gen['output']
                        teacher_loss_ = torch.mean(self.generative_model.crossentropy_loss(user_output_logp, y_input) *
                                                   torch.tensor(weight, dtype=torch.float32).cuda())
                        teacher_loss += teacher_loss_.cuda()
                        teacher_logit += user_result_give_gen['logit'] * torch.tensor(expand_weight,
                                                                              dtype=torch.float32).cuda()
                       # print(torch.tensor(expand_weight))
                       # print("user_result_give_gen",user_result_give_gen['logit'], user_result_give_gen['logit'].size())
                       # print("teacher_logit", teacher_logit)
                    student_output = student_model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                    # print('student output[logit]:', student_output['logit'].size())
                    # print('teacher output[logit]:', teacher_logit.size())
                    #print(F.log_softmax(student_output['logit'],dim=1), F.log_softmax(teacher_logit, dim=1))
                    # print(F.softmax(student_output['logit'],dim=1), F.softmax(teacher_logit, dim=1))
                    #student_loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1),
                                            # F.log_softmax(teacher_logit, dim=1))
                    # 使用log_softmax，student_loss报错为nan
                    # 实际使用后排查得出结论：预测结果需经过log_softmax(),理想结果只经过softmax()
                    student_loss = self.generative_model.kldiv_loss(F.log_softmax(student_output['logit'], dim=1),
                                                                    F.softmax(teacher_logit, dim=1))
                    # print(student_loss)
                    # if torch.isnan(student_loss):
                        #print("nan")
                        #break
                    loss = (self.ensemble_alpha * teacher_loss + self.ensemble_beta * student_loss +
                            self.ensemble_eta * diversity_loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 10)
                    self.generative_optimizer.step()
                    """for name, param in self.generative_model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.data).any():
                                print(name, "nan")
                            else:
                                print(name, "is not nan")
                                # print(name, param, param.grad)"""
                    #print(teacher_loss, student_loss, diversity_loss) # 输出的为当前生成器训练轮下，所有客户端的累加loss(不含diversity)
                    TEACHER_LOSS += self.ensemble_alpha * teacher_loss
                    STUDENT_LOSS += self.ensemble_beta * student_loss
                    DIVERSITY_LOSS += self.ensemble_eta * diversity_loss
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = update_generator(
                self.generator_K, self.student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
            for name, param in self.generative_model.named_parameters():
                # print(torch.max(param.grad))
                if torch.isnan(param).any():#any()用于检查可迭代对象中所有元素是否满足真值测试
                    print('server generator model param', name, 'has nan', param.size())
                if torch.isnan(param.grad).any():
                    print('server genertor model param.grad', name, "has nan", param.grad.size())
        TEACHER_LOSS = TEACHER_LOSS.cpu().detach().numpy()/self.generator_K * epoches
        STUDENT_LOSS = STUDENT_LOSS.cpu().detach().numpy()/self.generator_K * epoches
        DIVERSITY_LOSS = DIVERSITY_LOSS.cpu().detach().numpy()/self.generator_K * epoches
        info = ("Server Generator : Teacher Loss={:.4f}, Student Loss={:.4f}, Diversity Loss={:.4f}".format(
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS))
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()

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
