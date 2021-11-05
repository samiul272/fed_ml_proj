from argparse import ArgumentParser

from FedML.fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
import ssl
import argparse

from model import DiscriminatorDirectedModel

ssl._create_default_https_context = ssl._create_unverified_context


class DefaultArgs(argparse.ArgumentParser):
    def __init__(self):
        self.dataset_name = 'cifar10'
        self.data_dir = './../../../data/cifar10'
        self.partition_method = 'hetero'
        self.p_alpha = 0.5
        self.client_num_in_total = 10
        self.batch_size = 64
        self.data_loader = load_partition_data_cifar10
        super().__init__()


default_args = DefaultArgs()


def load_data(args=default_args, dataset_name=default_args.dataset_name):
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = args.data_loader(dataset_name, args.data_dir, args.partition_method,
                                 args.p_alpha, args.client_num_in_total, args.batch_size)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model():
    model = DiscriminatorDirectedModel()
    return model
