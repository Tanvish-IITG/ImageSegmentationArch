import torch
import torch.nn as nn
import torch.nn.functional as F

class PyPool(nn.Module):
    def __init__(self,h_kernels,v_kernels,ch,input_size):
        super().__init__()
        self.h_kernels = h_kernels
        self.v_kernels = v_kernels
        self.n = len(h_kernels)
        self.convlist1 = []
        self.convlist2 = []
        self.Upsample = []
        self.maxPool  = []
        self.avgPool  = []
        for i in range(self.n):
            self.convlist1.append(nn.Conv2d(ch,1,kernel_size=(3,3),padding='same'))
            self.convlist2.append(nn.Conv2d(ch,1,kernel_size=(3,3),padding='same'))
            self.Upsample.append(nn.Upsample(size=input_size,mode = "bilinear"))
            self.maxPool.append(nn.MaxPool2d(kernel_size=(h_kernels[i],v_kernels[i]), stride = ( h_kernels[i],v_kernels[i]), padding = 0))
            self.avgPool.append(nn.AvgPool2d(kernel_size=(h_kernels[i],v_kernels[i]), stride = ( h_kernels[i],v_kernels[i]), padding = 0))
        

    def forward(self,x:torch.Tensor):
        out = [x]
        for i in range(self.n):
            x_i_avg = self.avgPool[i](x)
            x_i_avg = self.convlist1[i](x_i_avg)
            x_i_avg = F.relu(x_i_avg)
            x_i_avg = self.Upsample[i](x_i_avg)
            x_i_max = self.maxPool[i](x)
            x_i_max = self.convlist2[i](x_i_max)
            x_i_max = F.relu(x_i_max)
            x_i_max = self.Upsample[i](x_i_max)
            out.append(x_i_avg)
            out.append(x_i_max)

        return torch.cat(out,dim = 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self,inp_ch):
        super().__init__()
        self.CA = ChannelAttention(inp_ch,ratio = 4)
        self.SA = SpatialAttention(kernel_size = 7)

    def forward(self,x:torch.Tensor):
        ca = self.CA(x)
        ca = ca * x
        sa = self.SA(ca)
        sa = sa * ca
        return sa



class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pypool = PyPool([2,4,8,16,32,64],[2,4,8,16,32,64],32,(512,512))
        self.ConvBlock1 = nn.Sequential(nn.Conv2d(44,32,3,padding="same"),
                            nn.ReLU(),
                            nn.Conv2d(32,32,3,padding="same"),
                            nn.ReLU())
        self.CBAM = CBAM(32)
        self.ConvBlock2 = nn.Sequential(nn.Conv2d(32,32,3,padding="same"),
                            nn.ReLU(),
                            nn.Conv2d(32,8,3,padding="same"),
                            nn.ReLU6())
                        
    def forward(self,x:torch.Tensor):
        x = self.pypool(x)
        x = self.ConvBlock1(x)
        x = self.CBAM(x)
        x = self.ConvBlock2(x)
        return x


if __name__ == "__main__":
    t = torch.rand((2,64,256,256))
    cm = AttentionBlock()
    y = cm(t)
    print(y.shape)


