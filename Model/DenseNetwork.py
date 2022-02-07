import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
      def __init__(self,inp_ch:int):
          super().__init__()
          self.conv1_1 = nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1)
          self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

          self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
          self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

          self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
          self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
          self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

          self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
          self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
          self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

          self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
          

      def forward(self,img):
        x = F.relu(self.conv1_1(img))
        x = F.relu(self.conv1_2(x))
        e1 = self.maxpool(x)
        x = F.relu(self.conv2_1(e1))
        x = F.relu(self.conv2_2(x))
        e2 = self.maxpool(x)
        x = F.relu(self.conv3_1(e2))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        e3 = self.maxpool(x)
        x = F.relu(self.conv4_1(e3))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        e4 = self.maxpool(x)
        
        return e1,e2,e3,e4

class DecoderBlock(torch.nn.Module):
      def __init__(self,input_channel,output_channel):
          super().__init__()
          self.deconv =  torch.nn.ConvTranspose2d(input_channel,output_channel,(2,2),stride = 2, padding = 0)
          self.conv = torch.nn.Conv2d(output_channel,output_channel,(3,3),padding = 'same')
          self.batchnorm = torch.nn.BatchNorm2d(output_channel)



      def forward(self,img):
          x = self.deconv(img)
          x = self.conv(x)
          x = self.batchnorm(x)
          x = torch.nn.functional.relu(x)

          return x

class Decoder(torch.nn.Module):
      def __init__(self,input_channel,mid_channels,output_channel):
            super().__init__()
            mid1,mid2,mid3 = mid_channels
            self.block1 = DecoderBlock(input_channel,mid1)
            self.block2 = DecoderBlock(2*mid1,mid2)
            self.block3 = DecoderBlock(2*mid2,mid3)
            self.block4 = DecoderBlock(2*mid3,output_channel)

      def forward(self, e):
            e1,e2,e3,e4 = e
            x = self.block1(e4)
            x = torch.cat((x,e3),dim = 1)
            x = self.block2(x)
            x = torch.cat((x,e2),dim = 1)
            x = self.block3(x)
            x = torch.cat((x,e1),dim = 1)
            x = self.block4(x)
            return x

class VGG16(torch.nn.Module):
      def __init__(self,inp_ch = 32,out_ch = 32):
            super().__init__()
            self.encoder = Encoder(inp_ch)
            self.decoder = Decoder(512,(256,128,64),out_ch)

      def forward(self, img):
            
            e = self.encoder(img)
            x = self.decoder(e)
            return x

if __name__ == "__main__":
    t = torch.rand((2,32,256,256))
    cm = VGG16()
    y = cm(t)
    print(y.shape)