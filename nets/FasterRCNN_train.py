import torch.nn as nn
import torch

from nets.classifier import Resnet50RoIHead, Resnet101RoIHead, VGG16RoIHead,Resnet50_FPNRoIHead
from nets.vgg16 import decom_vgg16
from nets.resnet50 import resnet50
from nets.resnet101 import resnet101
from nets.resnet50_FPN import resnet50_FPN
from nets.rpn import RegionProposalNetwork, resnet50_fpn_RPNhead



class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [4, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()  #对继承自父类的属性进行初始化，且用父类的初始化方法来初始化继承的属性
        self.feat_stride = feat_stride
        #---------------------------------#
        #   vgg和resnet50,resnet101,resnet50_FPN主干网络
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 7,
                spatial_scale   = 1,
                classifier      = classifier
            )

        elif backbone == 'resnet50':
            #   获得图像的特征层和分类层特征信息
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建建议框Proposal卷积网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )

        elif backbone=='resnet101':
            self.extractor, classifier = resnet101(pretrained)
            #---------------------------------#
            #   构建建议框Proposal卷积网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet101RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier)
            
        elif backbone=='resnet50_FPN':
            self.extractor, classifier = resnet50_FPN(pretrained)
            #---------------------------------#
            #   构建建议框Proposal卷积网络
            #---------------------------------#
            ratios = ratios*len(anchor_scales)
            self.rpn = resnet50_fpn_RPNhead(
                256, 256,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50_FPNRoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier)
    
    #x= [base_feature, img_size],在Suggestion_box.FasterRCNNTrainer.forward()产生
    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小
            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor.forward(x)

            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征,resnet50网络特征提取
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores
        elif mode == "fpn_head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #   取p2~p5层进行分类结果预测
            #---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature[:4], rois[:4], roi_indices[:4], img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
