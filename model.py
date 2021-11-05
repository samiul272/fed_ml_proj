import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorDirectedModel(nn.Module):
    def __init__(self):
        super(MNIST_gan, self).__init__()
        self.main_model = SlimmableModel()
        self.directing_discriminator = Discriminator()

    def get_main_model(self):
        return self.main_model

    def get_directing_discriminator(self):
        return self.directing_discriminator