from torch.optim import Optimizer

class vanilla_optimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        default = dict(lr=lr)
        super().__init__(params, default)

    def step(self, apply=True, lr=None, allow_unused=False):
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                if apply:
                    if lr is None:
                        p.data = p.data - p.grad.data * group['lr']
                    else:
                        p.data = p.data - p.grad.data * lr
        return grads

    def apply_grads(self, grads, beta=None, allow_unused=False):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                p.data = p.data - grads[i] * group['lr'] if beta == None else p.data - grads[i] * beta
                i += 1
        return


class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu)#创建字典，字典内值包括学习率，lambda，mu
        super(FedProxOptimizer,self).__init__(params,defaults)

    def step(self,vstar,closure=None):#vstar即为w*
        loss=None
        #closure应为一个函数，其作用为计算损失函数，返回损失值（即将迭代计算内容放在closure函数中）
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, pstar in zip(group['params'],vstar):#对于各模块参数，进行fedprox设定的优化公式进行更新
                #w<==== w-lr*(w'+lambda*(w-w*)+mu*w)#mu*w不知道哪里来的
                p.data=p.data - group['lr'] * (
                        p.grad.data+group['lamda']*(p.data-pstar.data.clone())+group['mu']*p.data)
        return group['params'], loss