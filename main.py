import argparse
from algorithms.servers.serverFedAG import FedAG
from algorithms.servers.serverFedAvg import FedAvg
from algorithms.servers.serverFedGen import FedGen
from algorithms.servers.serverFedFTG import FedFTG
from algorithms.servers.serverFedProx import FedProx
from utils.model_utils import create_model
import torch


def create_server(args, seed):
    model = create_model(args.model, args.dataset)
    if args.algorithm == "FedAG":
        server = FedAG(args, model, seed)
    elif args.algorithm == "FedAvg":
        server = FedAvg(args, model, seed)
    elif args.algorithm == "FedGen":
        server = FedGen(args, model, seed)
    elif args.algorithm == "FedFTG":
        server = FedFTG(args, model, seed)
    elif args.algorithm == "FedProx":
        server = FedProx(args, model, seed)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()  # 此行用于消除server显示赋值前可能引用
    return server


def run_job(args, i):
    torch.manual_seed(i+1)
    print('Start training iteration {}'.format(i))
    server = create_server(args, i)
    if args.train:
        server.train(args)
        server.test()


def main(args):
    for i in range(args.times):
        run_job(args, i)
        torch.cuda.empty_cache()
    print("finished training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Mnist-alpha0.1-ratio0.5")
    parser.add_argument('--model', type=str, default="cnn", choices=['cnn', 'resnet18'])  # 暂时不起作用
    parser.add_argument('--mode', type=str, default="weighted",
                        help="in fedgen,choices=[all,partial].in fedag,choices=[weighted,FTSG,ALA]")
    parser.add_argument('--train', type=bool, default=True, help="True for training, False for testing")
    parser.add_argument('--algorithm', type=str, default="FedAG")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gen_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--ensemble_lr', type=float, default=0.0001, help='only for FedGen')
    parser.add_argument('--lamda', type=float, default=1, help='Regularization term')
    parser.add_argument('--embedding', type=int, default=0, help='use embedding layer in generator network')
    parser.add_argument('--num_glob_iters', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=20)
    parser.add_argument('--num_users', type=int, default=20, help='number of users')
    parser.add_argument('--K', type=int, default=1,
                        help='number of iterations for generator or computation steps for fedgen')
    parser.add_argument('--times', type=int, default=3, help='run times')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--result_path', type=str, default='results', help='directory path to save results')

    args = parser.parse_args()

    print('-' * 40)
    print("Summary of training information:")
    print("Algorithm: {}".format(args.algorithm))
    print("Dataset: {}".format(args.dataset))
    print("Model: {}".format(args.model))
    print("Batch_size: {}".format(args.batch_size))
    print("Learning_rate: {}".format(args.learning_rate))
    print('Ensemble learning rate: {}'.format(args.ensemble_lr))
    print('number of users: {}'.format(args.num_users))
    print('number of global iterations: {}'.format(args.num_glob_iters))
    print('number of local epochs: {}'.format(args.local_epochs))
    print("Device: {}".format(args.device))
    print("-" * 40)
    main(args)
