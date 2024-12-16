import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_config import GENERATORCONFIGS

class Generator(nn.Module):
    def __init__(self, dataset='mnist', algorithm='FedGen', embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        self.algorithm = algorithm
        self.dataset = dataset
        self.embedding = embedding
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        if 'mnist' in self.dataset:
            self.img_size = 28
        elif 'CIFAR10' in self.dataset:
            self.img_size = 32
        else:
            raise ValueError(f"unsupported dataset")
        self.build_network() # 必须放在后面，否则build_network中不能识别self.img_size

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)  # 默认reduction='mean'
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def build_network(self):
        if self.algorithm == 'FedGen':
            if self.embedding:
                self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
            self.fc_layers = nn.ModuleList()
            for i in range(len(self.fc_configs)-1):
                input_dim, output_dim = self.fc_configs[i], self.fc_configs[i+1]
                print("Build layer {} X {} ".format(input_dim, output_dim))
                fc = nn.Linear(input_dim, output_dim)
                bn = nn.BatchNorm1d(output_dim)
                act = nn.ReLU()
                self.fc_layers += [fc, bn, act]
            self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
            self.representation_layer_subtitute = nn.Linear(self.fc_configs[-1], self.fc_configs[-1])
            print("Build last latent layer {} X {}".format(self.fc_configs[-1], self.latent_dim))
        else:
            self.init_size = self.img_size//4
            self.l1 = nn.Linear(100, 64*self.init_size**2) # 100 对应噪声维度
            self.l2 = nn.Linear(self.n_class, 64*self.init_size**2)

            self.conv_block0 = nn.BatchNorm2d(64*2)
            self.conv_block1 = nn.Sequential(nn.Conv2d(64*2, 64*2, 3, 1, padding=1),
                                nn.BatchNorm2d(64*2),
                                nn.LeakyReLU(0.2, inplace=True))
            self.conv_block2= nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, padding=1),
                               nn.BatchNorm2d(64),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Conv2d(64, 3 if self.dataset == 'CIFAR10' else 1, 3,1,1),
                               nn.Tanh(),
                               nn.BatchNorm2d(3 if self.dataset == 'CIFAR10' else 1, affine=False)
                            )

    def forward(self, labels, latent_layer_idx=-1, verbose=True): # y的数据选取不在此模型中涉及，包含噪声设计
        # latent_layer_idx 仅为训练FedGen预留
        if self.algorithm == 'FedGen':
            result = {}
            batch_size = labels.shape[0]
            eps = torch.rand((batch_size, self.noise_dim)).cuda()
            if verbose == True:
                result['eps'] = eps
            if self.embedding:
                y_input = self.embedding_layer(labels)
            else:
                y_input = torch.FloatTensor(batch_size, self.n_class)
                y_input.zero_()
                labels_int64 = labels.type(torch.LongTensor)
                # 转换为独热向量
                y_input.scatter_(1, labels_int64.view(-1,1), 1).cuda()
            z = torch.cat((eps.cuda(), y_input.cuda()),dim=1)
            for layers in self.fc_layers:
                z = layers(z)
            z = self.representation_layer(z)
            # z = self.representation_layer_subtitute(z)
            ### resnet18时为对齐矩阵维度
            result['output'] = z
            return result
        else:
            result = {}
            batch_size = labels.shape[0]
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            # print("non_y_hot",y_input)
            y_input.scatter_(1, labels.type(torch.LongTensor).view(-1, 1), 1)
            y_input = y_input.cuda()
            #print(y_input)
            nz = 100 if "cifar" in self.dataset.lower() or "mnist" in self.dataset.lower() else 256
            z = torch.randn(batch_size, nz, 1, 1).cuda()
            result['eps'] = z
            out_1 = self.l1(z.view(z.shape[0], -1))
            out_2 = self.l2(y_input.view(y_input.shape[0], -1))
            # print(out_2, out_2.size())
            out = torch.cat([out_1, out_2], dim=1)
            out = out.view(out.shape[0], -1, self.init_size, self.init_size)
            img = self.conv_block0(out)
            img = nn.functional.interpolate(img, scale_factor=2)
            img = self.conv_block1(img)
            img = nn.functional.interpolate(img, scale_factor=2)
            img = self.conv_block2(img)
            result['output'] = img
            return result

    def initialize(self): # xavier初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

class DiversityLoss(nn.Module):  # 所谓多样性损失
    """
    Diversity loss for improving the performance
    """

    def __init__(self, metric):
        """
        class initializer
        """
        super().__init__()  # 继承nn.Module()的__init__()
        self.metric = metric  # 采用何种方式计算张量距离，l1,l2,cosine
        self.cosine = nn.CosineSimilarity(dim=2)  # x*y/max(||x||*||y||,ε)，按指定维计算，计算后，该维度消失
        # 尚不清楚具体计算操作

    def computer_distance(self, tensor1, tensor2, metric):  # 计算两个tensor之间的距离，尚不清楚为什么在第2维度做距离计算
        """
        compute the distance between two tensors
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))  # 返回L1距离，即绝对值,这里在第2维取平均值，最终返回张量不含第2维
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))  # 返回L2距离，即平方差,这里在第2维取平均值，最终返回张量不含第2维
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)  # 返回余弦距离，为0时完全重合，1时正交，2时完全相反。

    def pairwise_distance(self, tensor, how):  # 计算张量行之间的成对距离
        """
        compute the pairwise distances between a tensor's rows
        """
        n_data = tensor.size(0)  # 获取张量首维的长度
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))  # 对原张量维度中为1的维进行扩展，扩展成指定形状
        tensor2 = tensor.unsqueeze(dim=1)  # 插入第1维
        return self.computer_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):  # layer即为生成器输出
        """
        Forward Propagation
        """
        if len(layer.shape) > 2:  # 如果生成张量维度超过2维，则进行reshape
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)  # 返回张量形状为(n_data,n_data)
        noise_dist = self.pairwise_distance(noises, how='l2')  # 返回l2损失，张量形状为(n_data,n_data)
        return torch.exp(torch.mean(-noise_dist * layer_dist))  # torch.mean()不指定维度(dim)返回所有元素的平均值

