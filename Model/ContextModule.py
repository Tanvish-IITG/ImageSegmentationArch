import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBasicBlock(nn.Module):
    def __init__(self,kernel,input_ch,output_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch,output_ch,kernel,padding="same")
        self.conv2 = nn.Conv2d(output_ch,output_ch, kernel,padding="same")
        self.conv3 = nn.Conv2d(output_ch,output_ch, kernel,padding="same")

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class ContextModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvBlock1 = ConvBasicBlock(3,3,8)
        self.ConvBlock2 = ConvBasicBlock(5,3,8)
        self.ConvBlock3 = ConvBasicBlock(7,3,8)
        self.ConvBlock4 = ConvBasicBlock(9,3,8)

    def forward(self,x:torch.Tensor):
        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x)
        x3 = self.ConvBlock3(x)
        x4 = self.ConvBlock4(x)
        out = torch.cat([x1,x2,x3,x4], dim = 1)
        return out

if __name__ == "__main__":
    t = torch.rand((2,3,32,32))
    cm = ContextModule()
    y = cm(t)
    print(y.shape)