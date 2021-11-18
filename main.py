from argparse import ArgumentParser

import setproctitle
import wandb

from FedML.fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
import ssl
import argparse

from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_experiments.distributed.fedall.main_fedavg import init_training_device
from fedml_experiments.standalone.fedavg.main_fedavg import custom_model_trainer

ssl._create_default_https_context = ssl._create_unverified_context


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--model', type=str, default='resnet56', metavar='N',
    #                     help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='FedML/data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')
    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")
    parser.add_argument(
        "--is_mobile", type=int, default=0, help="whether the program is running on the FedML-Mobile server side"
    )
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=20, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_cifar10(dataset_name, args.data_dir, args.partition_method,
                                            args.partition_alpha, args.client_num_in_total, args.batch_size)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


comm, process_id, worker_number = FedML_init()
wandb.init(project='samiul_fedavg_testing', name='fedavg_testing')
parser = argparse.ArgumentParser()
args = add_args(parser)

dataset = load_data(args, 'cifar10')
train_data_num, test_data_num, train_data_global, test_data_global, \
train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
model = MobileNetV3(model_mode="LARGE")
setproctitle.setproctitle("test_fed")
device = init_training_device(process_id, worker_number - 1, 3)

model_trainer = custom_model_trainer(args, model)
fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
fedavgAPI.train()
