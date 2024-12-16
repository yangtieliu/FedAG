from algorithms.users.userFedProx import UserFedProx
from algorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
#Implementation for FedProx Server

class FedProx(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)
        self.init_loss_fn()

        #Initialize data for all users
        self.data=read_data(args.dataset)
        #返回客户端列表，groups（列表）,（后面都是字典）train_data,test_data,proxy_data
        #总用户数量
        total_users=len(self.data[0])
        #打印总用户数量
        print("Users in total:{}".format(total_users))

        #在总用户数量循环中
        for i in range(total_users):
            #按序获取不同客户端的id,训练集和测试集
            id, train_data, test_data=read_user_data(i,self.data)
            #创建FedProx客户端对象
            user=UserFedProx(args,id,model,train_data,test_data, lamda=self.lamda, use_adam=False)
            #添加该对象至self.users列表中
            self.users.append(user)
            #更新总训练样本数
            self.total_train_samples += user.train_samples

        #打印激活客户端占比，结束创建FedProx服务器
        print("Number of users/total users:", self.num_users,"/",total_users)
        print("Finished creating FedProx server")

    def train(self,args):#定义服务器训练函数
        #在全局通信轮内
        for glob_iter in range(self.num_glob_iters):
            #打印当前通信轮
            print("\n\n-------------Round number:",glob_iter,"-----------\n\n")
            #选择客户端根据激活客户端数量，与当前通信轮并无关系
            self.selected_users=self.select_users(self.num_users, way="random")
            #将当前全局模型参数传递给所有客户端（包括未激活客户端）
            self.send_parameters()
            #评估当前模型（打印当前全局精度及损失）
            self.evaluate()
            #允许被选客户端进行训练
            for user in self.selected_users:#allow selected users to train
                user.train(glob_iter)#glob_iter在客户端训练函数中只用于学习率更新策略
            self.aggregate_parameters(partial=False)#按比例（客户端样本占比）聚合模型参数\
            self.evaluate_server(self.data, True)
        self.save_results(args)#保存实验数据
        self.save_model()#保存最终全局模型