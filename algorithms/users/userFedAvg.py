import copy
import torch
from algorithms.users.userbase import User
import numpy as np


class UserAvg(User):
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        if args.device == 'cuda':
            self.cuda = True
        self.model_before = copy.deepcopy(self.model)
    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        self.model.train()
        self.model_before = copy.deepcopy(self.model)
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            for i in range(
                    int(np.ceil(self.train_samples/self.batch_size))  #self.K
                           ):
                result = self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                #if self.model_name == 'resnet18':
                    #output = self.model(X)
                    #output = torch.nn.functional.log_softmax(output, dim=1)
                #else:
                output = self.model(X)['output']
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        if lr_decay:
            self.lr_scheduler.step()
        test_acc, _, num_test_samples = self.test()
        test_acc = test_acc / num_test_samples
        print("client test_acc", test_acc)

    def compare_param(self):
        for client_model_param, client_model_before_param in zip(self.model.parameters(),
                                                                 self.model_before.parameters()):
            if torch.sum(client_model_param - client_model_before_param) == 0:
                print("client_param == client_before_param")
            else:
                print("client_param != client_before_param")