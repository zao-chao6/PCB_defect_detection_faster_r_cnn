import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool,MultiScaleRoIAlign,RoIAlign
from collections import OrderedDict

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

# faster rcnn 网络部分的分类
class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #   in_features:2048 -> out_features:n_class * 4
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class )
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        # 对特征层的建议框进行缩放
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #   得到一个将预选框映射到一个预测特征层
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc.view(fc.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


class Resnet50_FPNRoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50_FPNRoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的结果进行全连接
        #   最大池化层
        #--------------------------------------#
        self.maxpool =nn.AdaptiveMaxPool2d(7)   # output size = (1, 1)
        self.fc = nn.Linear(12544, 1024)
        #--------------------------------------#
        #   对ROIPooling后的结果进行回归预测
        #   in_features:256 -> out_features:n_class * 4
        #--------------------------------------#
        self.cls_loc = nn.Linear(1024, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(1024, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.fc, 0, 0.001)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = MultiScaleRoIAlign(featmap_names=['p2', 'p3', 'p4', 'p5'], output_size=7, sampling_ratio=2, canonical_scale=600, canonical_level=4)
        # self.roi = RoIAlign(output_size=7, sampling_ratio=2, spatial_scale=1)

    def forward(self, x, rois, roi_indices, img_size):
        # import time
        # start=time.time()
        # 将列表转换为OrderedDict类型，MultiScaleRoIAlign（）数据准备
        Ordered_x = OrderedDict(p2=x[0],p3=x[1],p4=x[2],p5=x[3])
        # img_size_p1 = [(img_size[0], img_size[1])]
        rois_p = []
        # 将每层背景和前景的建议框进行合并
        for p in range(len(x)):
            n, _, _, _ = x[p].shape
            if x[p].is_cuda:
                roi = rois[p].cuda()
                rois_p.append(torch.flatten(roi, 0, 1))
            else:
                rois_p.append(torch.flatten(rois[p], 0, 1))
            

        # 将p2~p4层Proposal框合并为一个[tensor]
        rois_x = [torch.cat(rois_p, dim=0)]
        # 使用MultiScaleRoIAlign（）在多尺度特征层进行预测
        pool = self.roi(Ordered_x, rois_x, [img_size])
        #--------------------------------------------------------------#
        # 将 rois 的信息在第一维度上进行展平操作
        # 当输入为一张图片的时候，这里获得的f7的shape为[1024, 12544]
        #--------------------------------------------------------------#
        fc=self.maxpool(pool)
        fc = pool.view(pool.size(0), -1)
        # 进行全链接层。最终输出 1024 维的特征向量。
        fc7 = self.fc(fc)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))

        # end=time.time()-start
        # print("代码运行时间：",end)
        return roi_cls_locs, roi_scores


class Resnet101RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet101RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
