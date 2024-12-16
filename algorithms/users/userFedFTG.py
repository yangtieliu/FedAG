import torch
import torch.nn.functional as F
import numpy as np
import copy
from algorithms.users.userbase import User
from utils.model_utils import create_generator

class UserFedFTG(User):
    def __init__(self, args, id, model,  # generative_model,
                 train_data, test_data, available_labels, label_info, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        # self.gen_batch_size =args.gen_batch_size
        # self.generative_model = generative_model
        self.available_labels = available_labels
        self.label_info = label_info
        self.gpu(args)

    def gpu(self, args):
        if args.device == "cuda":
            self.cuda = True
            self.model.cuda()

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += int(count)

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:0 for label in range(self.unique_labels)}

    def train(self, verbose=True):
        self.clean_up_counts()
        self.model.train()
        for epoch in range(self.local_epochs):
            for i in range(int(np.ceil(self.train_samples/self.batch_size))):  # 不同于其他方法中的训练迭代次数

                samples = self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                self.update_label_counts(samples['labels'], samples['counts'])
                result = self.model(X)
                y_pred = result['output']
                loss = self.loss(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
        if verbose:
            test_acc, test_loss, samples_num = self.test()
            print("Client test_acc: {}".format(test_acc/samples_num))
            print("Client test_loss: {}".format(test_loss))
            self.lr_scheduler.step()


