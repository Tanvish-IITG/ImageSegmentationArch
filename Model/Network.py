import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.AttentionBlock import AttentionBlock
from Model.ContextModule import ContextModule
from Model.DenseNetwork import VGG16

class Network(nn.Module):
    def __init__(self,size:int, classes:int):
        super().__init__()
        self.cm = ContextModule()
        self.vgg = VGG16(35,32)
        self.ab = AttentionBlock(size,classes)

    def forward(self,x:torch.Tensor):
        x = self.cm(x)
        x = self.vgg(x)
        x = self.ab(x)
        return x

if __name__ == "__main__":
    t = torch.rand((2,3,256,256))
    net = Network(256)
    y = net(t)
    print(y.shape)