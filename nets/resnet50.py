import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #1*1的卷积压缩通道数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #3*3卷积特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #1*1复原通道数 

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)   #判断是否有残差边，有残差边即为：输入维度和输出维度发生改变，对应conv block
                                            #无残差边：输入维度=输出维度，对应identity block

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, include_top=True,num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.include_top = include_top
        self.inplanes = 64
        super(ResNet, self).__init__()

        # input（600,600,3） -> conv2d stride（300,300,64）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#输入3通道，卷积核大小kernel_size=7*7，
                                                                                     #步长stride=2，输出通道数=64，bias偏移量
        self.bn1 = nn.BatchNorm2d(64)      #标准化（归一化）
        self.relu = nn.ReLU(inplace=True)  #激活函数

        # 300,300,64 -> 150,150,64  最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AvgPool2d(7)  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        #resnet模型每层进行参数学习，如：layer1中每层进行模型训练
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                new_var = 1
                n = m.kernel_size[0] * m.kernel_size[new_var] * m.out_channels  #通道数的改变（如：256->64）
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)          #更改resnet50网络中每层中权重数据
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample（下采样）
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # resnet50网络层数堆积，layer=[3, 4, 6, 3]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            x = x.view(x.size(0), -1)  # 传入神经网络之前将tensor变形，
            x = self.fc(x)  # 输入全连接层,神经网络输入准备
        
        return x

def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])  #对应resnet50的网络结构shape，第五次压缩是在roi中使用，有3个bottleneck。
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    print('features:', features)
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    print('classifier:', classifier)
    
    features    = nn.Sequential(*features)
    print('features:', features)
    classifier  = nn.Sequential(*classifier)
    print('classifier:', classifier)
    return features, classifier


# net = ResNet(Bottleneck, [3, 4, 6, 3])
# print(net)

