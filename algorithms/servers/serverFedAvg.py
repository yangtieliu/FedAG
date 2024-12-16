from algorithms.users.userFedAvg import UserAvg
from algorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import time
import torch
import copy
import os

class FedAvg(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        self.data = read_data(args.dataset)
        total_users = len(self.data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, self.data, count_labels=True)
            user = UserAvg(args, id, model, train_data, test_data, use_adam=self.use_adam)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.init_loss_fn()
        print("Number of users/total users:", args.num_users, "/", total_users)
        print("Finished creating FedAvg server.")

    def save_temp_model(self, glob_iter):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server_" + str(glob_iter) + ".pt"))

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n------------Round Number: ", glob_iter, "---------\n\n")
            self.selected_users, user_idxs = self.select_users(args.num_users, way="random", return_idx=True)
            print("selected users", user_idxs)
            self.send_parameters(mode="all")
            self.evaluate()
            self.time_start = time.time()
            server_model_before = copy.deepcopy(self.model)
            """for client_param, server_before_param in zip(self.selected_users[0].model.parameters(), server_model_before.parameters()):
                if torch.sum(client_param - server_before_param) == 0:
                    print("client_param == server_before_param")
                else:
                    print("client_param != server_before_param")"""
            for user in self.selected_users:
               # model_before = copy.deepcopy(user.model)
                user.train(glob_iter, lr_decay=True, count_labels=True)
                # print(user.model)
               # for client_model_param, client_model_before_param in zip(self.selected_users[0].model.parameters(),
                                                                        # model_before.parameters()):
                    #if torch.sum(client_model_param - client_model_before_param) == 0:
                      #  print("client_param == client_before_param")
                   #else:
                     #   print("client_param != client_before_param")
            self.time_end = time.time()
            train_time = (self.time_end - self.time_start)/len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            # server_model_before_train = copy.deepcopy(self.model)
            self.aggregate_parameters(partial=False)
            """for server_param, server_before_param in zip(self.model.parameters(), server_model_before.parameters()):
                # print(server_param)
                if torch.sum(server_param - server_before_param) == 0:
                    print("server_param == server_before_param")
                else:
                    print("server_param != server_before_param")"""
            # self.model.eval()
            #for param in self.model.parameters():
                #print("param")
                #print(param.grad)

            self.evaluate_server(self.data, save=True)

        self.save_results(args)
        self.save_model()
