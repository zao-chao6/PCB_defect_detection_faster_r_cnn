import math
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4   #通道倍增数

    def __init__(self, inplanes, planes, stride=1, downsample=None,k_size=3):
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

        # 激活+下采样
        self.relu = nn.ReLU(inplace=True)
        # 加入ECA模型
        self.eca = eca_layer(planes * 4, k_size)

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
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)   #判断是否有残差边，有残差边即为：输入维度和输出维度发生改变，对应conv block
                                            #无残差边：输入维度=输出维度，对应identity block
        out += residual
        out = self.relu(out)

        return out

class ResNet50_FPN(nn.Module):
    def __init__(self, block, layers, num_classes=100,k_size=[1, 1, 1, 1]):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        super(ResNet50_FPN, self).__init__()
        self.inplanes = 64

        #处理输入的C1模块（C1代表了RestNet的前几个卷积与池化层）
        # input（600,600,3） -> conv2d stride（300,300,64）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#输入3通道，卷积核大小kernel_size=7*7，
                                                                                     #步长stride=2，输出通道数=64，bias偏移量
        self.bn1 = nn.BatchNorm2d(64)      #标准化（归一化）
        self.relu = nn.ReLU(inplace=True)  #激活函数

        # 300,300,64 -> 150,150,64  最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        ''' Bottom-up layers ,搭建自下而上的C2，C3，C4，C5'''
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0],int(k_size[0]))
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1],int(k_size[1]), stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2],int(k_size[2]), stride=2)
        # 38,38,1024 -> 19,19,2048 
        self.layer4 = self._make_layer(block, 512, layers[3],int(k_size[3]), stride=2)
        
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
        # Smooth layers,3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers,横向连接，保证通道数相同
        self.latlayer3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # 19,19,p5 ->10,10, p6  最大池化
        self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0, ceil_mode=True)

        # 最池化层和全连接层
        self.maxpool1 = nn.AdaptiveMaxPool2d(7)  # output size = (1, 1)
        # self.fc = nn.Linear(256, 256)
        
        #resnet模型每层进行参数学习，如：layer1中每层进行模型训练
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks,k_size,  stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample（下采样）
        #   将输入的downsample（x）自动按照Sequential（）里面的布局，顺序执行，
        #   目的：优化类似于这种结构：x = self.bn1(x)，x = self.relu(x)，降低运行内存。
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,downsample,k_size))
        self.inplanes = planes * block.expansion
        # resnet50网络层数堆积，layer=[3, 4, 6, 3]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,k_size))
        return nn.Sequential(*layers)

    #   通过上采样后，进行特征融合
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return nn.functional.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        # 自己构建的fpn网络，c1~c4层搭建
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down 降通道数
        p5 = self.toplayer(c5)
        #   upsample
        p4 = self._upsample_add(p5, self.latlayer3(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        
        # Smooth,特征提取，卷积的融合，平滑处理
        p5 = self.smooth4(p5)
        #   19,19,256->10,10,256  经过maxpool得到p6，用于rpn网络中
        p6 = self.maxpool_p6(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)
        
        x = [p2, p3, p4,p5, p6]
        #   对fpn的特征层进行全连接层
        # for key,value in x.items() :
        #     value  = self.avgpool(value)
        #     # view()函数的功能根reshape类似，用来转换size大小。x = x.view(batchsize, -1)中batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        #     value = value.view(value.size(0), -1)
        #     # value = torch.flatten(value, 1) #flatten(x,1)是按照x的第1个维度拼接（按照列来拼接，横向拼接）；flatten(x,0)是按照x的第0个维度拼接（按照行来拼接，纵向拼接）
        #     value = self.fc(value)
        #     # value = value.view(-1)
        #     x.update(key,value)
        
        return x

# test
# FPN = ResNet50_FPN(Bottleneck, [3, 4, 6, 3])
# print('FPN:',FPN)


def resnet50_FPN(pretrained=False):
    # 对应resnet50的网络结构shape，第五次压缩是在roi中使用，有3个bottleneck。
    model = ResNet50_FPN(Bottleneck, [3, 4, 6, 3])
    # print('ResNet50_FPN:',model)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.smooth1(p4层)，获得多个p2, p3, p4, p5,p6不同尺度的特征层
    #----------------------------------------------------------------------------#
    # features = list([model.conv1, model.bn1, model.relu,model.maxpool, model.layer1, model.layer2, model.layer3,model.layer4, 
                    # model.toplayer,model.smooth4, model.smooth3, model.smooth2, model.smooth1])
    
    # features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1,
    #                 model.layer2, model.layer3, model.layer4, ])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.smooth3（p2）到model.toplayer（p5）特征层
    #----------------------------------------------------------------------------#
    classifier = list([model.smooth1, model.smooth2, model.smooth3, model.smooth4,
                       model.maxpool1])
    
    # 特征提取（feature map）
    features = model
    # features    = nn.Sequential(*features)      # 函数参数（位置参数，*可变参数（以tuple/list形式传递），**关键字参数（以字典形式传递），
                                                # 默认参数（需要放在参数中最右端，避免传参是歧义））
    print('features:', features)
    classifier  = nn.Sequential(*classifier)    #在进行完roipool层后，进行回归和分类预测
    print('classifier:', classifier)
    return features, classifier

