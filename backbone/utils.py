
import torch
import torch.nn as nn
import torch.nn.functional as F

def xdog_func(input, params):
    # params is output of sigmoid (0-1) in format of (sigma, k, tau)
    ba, ch, fh, fw = input.shape
    max_sigma = 1.0
    sigma = params[:,:,0] * max_sigma + 0.01
    k = params[:,:,1] + 1.0 # k in range [1,2]
    rou = torch.zeros_like(sigma)
    tau = torch.unsqueeze(params[:,:,2], 2)
    
    sigma_1 = sigma
    sigma_2 = sigma * k
    out_1 = smooth_func(input, sigma_1, sigma_1, rou)
    out_2 = smooth_func(input, sigma_2, sigma_2, rou)
    
    out = out_1.view(ba,ch,-1) - tau*out_2.view(ba,ch,-1)
    return out.view(ba,ch,fh,fw)
            

def smooth_func(input, sigma_x, sigma_y, rou):
    ba, ch, fh, fw = input.shape
    kw = 5 # odd number
    # max_sigma = (kw - 1) / 2

    x, y = torch.meshgrid(torch.linspace(-int(kw/2),int(kw/2),kw), torch.linspace(-int(kw/2),int(kw/2),kw))
    x = x.repeat(ba,ch,1,1).type_as(input)
    y = y.repeat(ba,ch,1,1).type_as(input)

    sigma_x = sigma_x.view(ba,ch,1,1)
    sigma_y = sigma_y.view(ba,ch,1,1)
    rou = rou.view(ba,ch,1,1)

    u = x/sigma_x
    v = y/sigma_y

    gs = torch.exp(-(u*u+v*v-2*rou*u*v)/(2*(1-rou*rou)))
    gs = gs/gs.sum(dim=(2,3), keepdim=True) # normalize
    ks = gs.view(ba*ch, 1, kw, kw)

    out1n = F.conv2d(input.view(1,-1,fh,fw), ks, padding=int(kw/2), groups=ba*ch)
    return out1n.view(ba,ch,fh,fw)
    
class sigmaBranch(nn.Module):
    def __init__(self, in_planes, ratios=1.0/8):
        super(sigmaBranch, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
        out_planes = 3*in_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 1, bias=True)

    def forward(self, x):
        ba, ch, fh, fw = x.shape
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(ba,ch,3)
        return torch.sigmoid(x)

class fusionBranch(nn.Module):
    def __init__(self, in_planes, ratios=1.0/8):
        super(fusionBranch, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
        out_planes = 2*in_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 1, bias=True)

    def forward(self, x):
        ba, ch, fh, fw = x.shape
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(ba,ch,2)
        return F.softmax(x, 2)

class ordinaryConvs(nn.Module):
    def __init__(self, in_planes):
        super(ordinaryConvs, self).__init__()
        hidden_planes = int(in_planes/2)
        out_planes = in_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    
class SENet(nn.Module):
    def __init__(self, in_planes, ratios=1.0/8):
        super(SENet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
        out_planes = in_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 1, bias=True)

    def forward(self, x):
        ba, ch, fh, fw = x.shape
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class XDogNet(nn.Module):
    def __init__(self, in_planes, ratios=1.0/8):
        super(XDogNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes*ratios)+1
        out_planes = 3*in_planes
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, out_planes, 1, bias=True)

    def forward(self, x):
        ba, ch, fh, fw = x.shape
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(ba,ch,3)
        return torch.sigmoid(x)
