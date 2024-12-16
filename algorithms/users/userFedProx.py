from algorithms.users.userbase import User
from algorithms.optimizers.optimizer import FedProxOptimizer
import torch

class UserFedProx(User):#FedProx客户端算法类
    def __init__(self,args,id,model,train_data,test_data,lamda, use_adam=False,):
        super().__init__(args,id,model,train_data,test_data,use_adam=use_adam)
        #除从userbase继承的属性外，还需要额外添加以下属性：
        #定义fedProx的优化器与学习率调整策略（指数衰减，指数为0.99）
        #学习率调整策略定义多余，User类中已定义
        self.optimizer = FedProxOptimizer(self.model.parameters(),lr=self.leaning_rate,lamda=lamda)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,gamma=0.99)
        if args.device=='cuda':
            self.cuda=True
    def update_label_counts(self,labels,counts):#更新标签计数，同其他算法定义
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):#清空标签计数，同其他算法定义
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter,lr_decay=True,count_labels=False):#定义客户端训练函数
        self.clean_up_counts()#首先清除标签计数
        self.model.train()#设定模型为训练模式
        #cache global model initialized value to local model
        #将全局模型的参数缓存到本地模型
        # 即将self.model.parameters()的模型参数逐层复制到self.local_model(作为w*)
        self.clone_model_parameters(self.model.parameters(), self.local_model)
        #在本地训练轮中
        for epoch in range(self.local_epochs):
            #模型为训练模式
            self.model.train()
            #在所有batch中
            for i in range(self.K):
                #获取一个训练batch
                result=self.get_next_train_batch(count_labels=count_labels)
                #分离result字典中的数据和标签
                X,y=result['X'],result['y']
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                #如果记录标签数量为真，则更新标签计数
                if count_labels:
                    self.update_label_counts(result['labels'],result['counts'])
                #优化器为零梯度
                self.optimizer.zero_grad()
                #输出模型预测
                output=self.model(X)['output']
                #计算输出损失
                loss=self.loss(output,y)
                #反向传播
                loss.backward()
                #更新优化器,参数为w*,由FedProx优化器定义
                self.optimizer.step(self.local_model)

        if lr_decay:
            self.lr_scheduler.step(glob_iter)
