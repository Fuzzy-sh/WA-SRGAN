import torch
from torchvision import models, datasets
from torch import nn,optim
from collections import OrderedDict
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import math
from math import exp
from torch import nn,optim
from torch.nn import SyncBatchNorm
from torch.nn.utils import weight_norm



####################################################################### 
# class upsample  
#######################################################################     
class UpsampleBlocks_wdsr(nn.Module):
      def __init__(self, channels, upscale):
        super(UpsampleBlocks_wdsr,self).__init__()
        self.conv1=nn.Conv2d(channels, channels*upscale**2,kernel_size=3,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(upscale)
        self.relu=nn.PReLU()
      def forward(self,x):
        x=self.conv1(x)
        x=self.pixel_shuffle(x)
        x=self.relu(x)
        return x
        
        
####################################################################### 
#Discriminator_WGAN_GP without batchnormalization .. and using Leakyrelue (1)
####################################################################### 
class Discriminator_Self_Attention_WGAN_WN (nn.Module):
      def __init__(self):
    
        super(Discriminator_Self_Attention_WGAN_WN, self).__init__()
        layer1=[]
        layer2=[]
        layer3=[]
        layer1.append(nn.Conv2d(3,64, kernel_size=3, padding=1))
        layer1.append(nn.PReLU())
    

        layer1.append(weight_norm(nn.Conv2d(64,64, kernel_size=3, padding=1, stride=2)))
        layer1.append(nn.PReLU())
      
  
        layer1.append(weight_norm(nn.Conv2d(64,128, kernel_size=3, padding=1)))
        layer1.append(nn.PReLU())
    

        layer1.append(weight_norm(nn.Conv2d(128,128, kernel_size=3, padding=1, stride=2)))
        layer1.append(nn.PReLU())

        layer1.append(weight_norm(nn.Conv2d(128,256, kernel_size=3, padding=1)))
        layer1.append(nn.PReLU())
    

        layer1.append(weight_norm(nn.Conv2d(256,256, kernel_size=3, padding=1, stride=2)))
        layer1.append(nn.PReLU())
    
        layer2.append(Self_Attention(256))
    

        layer3.append(weight_norm(nn.Conv2d(256,512, kernel_size=3, padding=1)))
        layer3.append(nn.PReLU())
    

        layer3.append(weight_norm(nn.Conv2d(512,512, kernel_size=3, padding=1, stride=2)))
        layer3.append(nn.PReLU())
    
        layer3.append(nn.AdaptiveAvgPool2d(1))
        layer3.append(nn.Conv2d(512,1024, kernel_size=1))
        layer3.append(nn.PReLU())
        layer3.append( nn.Conv2d(1024,1, kernel_size=1))
    
        self.layer1=nn.Sequential(*layer1)
        self.layer2=nn.Sequential(*layer2)
        self.layer3=nn.Sequential(*layer3)
    
      def forward(self, x):
        batch_size=x.size()[0]
        layer1=self.layer1(x)
        layer2,atten=self.layer2(layer1)
        layer3=self.layer3(layer2)
        #Removed the sigmoid activation from the last layer of the discriminator
        return layer3.view(batch_size)


##############################################################
# ResNet_wdsr Blocks        
#############################################################
class ResBlock_wdsr_WN(nn.Module):
  def __init__(self, n_feats, expantion_ratio,res_scale=1.0, low_rank_ratio=0.8):
    super(ResBlock_wdsr_WN,self).__init__()
    self.res_scale=res_scale
    self.module=nn.Sequential(
        weight_norm(nn.Conv2d(n_feats,n_feats*expantion_ratio,kernel_size=1)),
        nn.ReLU(inplace=True),
        weight_norm(nn.Conv2d(n_feats*expantion_ratio, int(n_feats*low_rank_ratio), kernel_size=1)), 
        weight_norm(nn.Conv2d(int(n_feats*low_rank_ratio), n_feats , kernel_size=3, padding=1))      

    )
  def forward(self, x):
    return x+self.module(x)
    


#######################################################################  
#Generator_WGAN_GP with one layer of SelfAttention with 8 residual 
####################################################################### 
class Generator_Self_Attention_WGAN_WDSR_WN (nn.Module):
      def __init__(self, scale_factor):
        super(Generator_Self_Attention_WGAN_WDSR_WN,self).__init__()
        num_upsample_block=1
        if scale_factor>2:
            num_upsample_block=int(math.log(scale_factor,2))
        num_res_wdsr_block=8
        
        self.block1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9,padding=4),
            nn.PReLU()
        )
        Resblock_wdsr=[ResBlock_wdsr_WN(64,4) for _ in range(num_res_wdsr_block)]
        self.Resblock_wdsr=nn.Sequential(*Resblock_wdsr)

        self.block10=nn.Sequential(
             weight_norm(nn.Conv2d(64,64,kernel_size=5,padding=2))
   
        )
        self.self_atten=Self_Attention(64)

        block11=[UpsampleBlocks_wdsr(64,2) for _ in range(num_upsample_block)]
        block11.append(nn.Conv2d(64,3, kernel_size=9, padding=4))
        self.block11=nn.Sequential(*block11)

      def forward(self,x):
        block1=self.block1(x)
        Resblock_wdsr=self.Resblock_wdsr(block1)
        block10=self.block10(Resblock_wdsr)
        FSA_layer,atten2=self.self_atten(block1 + block10)
        # FSA_layer2,atten2=self.self_atten2(FSA_layer1)
        block11=self.block11(FSA_layer)
        output=block11
        return (torch.tanh(output)+1)/2

    
####################################################################### 
# GeneratorLoss  
####################################################################### 

class GeneratorLoss(nn.Module):
    def __init__(self,state_dict,model):
        super(GeneratorLoss, self).__init__()
        vgg = models.vgg19(pretrained=False)
        classifier= nn.Sequential(OrderedDict([
                    ('0', nn.Linear(25088,4096)),
                    ('1',nn.ReLU(inplace=True)),
                    ('2', nn.Dropout(p=0.5)),
                    ('3', nn.Linear(4096,4096)),
                    ('4',nn.ReLU(inplace=True)),
                    ('5', nn.Dropout(p=0.5)),
                    ('6', nn.Linear(4096,2)),
                    ('output', nn.LogSoftmax(dim=1))
                                                 ]))
        vgg.classifier=classifier
        vgg.eval()
        
        #load vgg model that is trained by BC images
      
        vgg.load_state_dict(state_dict)
        loss_network = nn.Sequential(*list(vgg.features)[:37]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.marginal_loss=nn.MarginRankingLoss()
        self.tv_loss = TVLoss()
        self.l1_loss=nn.L1Loss()
        self.model=model

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -1*out_labels
        
        # Perception Loss/ vgg-loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Image Loss/mse-loss
        image_loss = self.mse_loss(out_images, target_images)
        
        # tv loss
        tv_loss = self.tv_loss(out_images)
        return image_loss  + 1e-3 * adversarial_loss +  2e-6 * perception_loss+ 2e-8 * tv_loss
        


####################################################################### 
# calc_gradient_penalty  
####################################################################### 
def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.randn (real_data.size(0),1,1,1)
    alpha = alpha.to(device)
      
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    d_interpolates = netD(interpolates)
    fake = torch.ones(d_interpolates.size())
    fake = fake.to(device)
      
    gradients = torch.autograd.grad(
          outputs=d_interpolates,
          inputs=interpolates,
          grad_outputs=fake,
          create_graph=True,
          retain_graph=True,
          only_inputs=True,
      )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty		


####################################################################### 
# l2 loss that is introduced as TV loss as well
####################################################################### 

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    
####################################################################### 
# Self_Attention  
#######################################################################     
class Self_Attention (nn.Module):
      def __init__(self, in_dim):
        super(Self_Attention,self).__init__()
        self.query_conv=nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv=nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv=nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
      def forward(self,x):
        B,C,W,H=x.size()
        query=self.query_conv(x).view(B, -1, W*H)
        query=query.permute(0,2,1)
        key=self.key_conv(x).view((B, -1, W*H))
        energy = torch.bmm(query, key)
        attention=self.softmax(energy).permute(0,2,1)
        value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(value, attention).view(B, C, W,H)
        out= self.gamma*out+x
        return out, attention

