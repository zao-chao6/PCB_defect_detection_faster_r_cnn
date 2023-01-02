
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox
from collections import OrderedDict

class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 1000,  
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 1000, 
        min_size            = 16
    
    ):
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #-----------------------------------#
        #   预测用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #-----------------------------------#
        #   将先验框转换成tensor
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)
        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)
        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size    = self.min_size * scale
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]
        score       = score[keep]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi     = roi[order, :]
        score   = score[order]

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]
        return roi


class resnet50_fpn_RPNhead(nn.Module):
    def __init__(
        self,
        in_channels=512,
        mid_channels=512,
        ratios=[0.5, 1, 2],
        anchor_scales=[4, 16, 32],
        feat_stride=16,
        mode="training",
    ):
        super(resnet50_fpn_RPNhead, self).__init__()
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        #每个网格上默认的先验框数量
        n_anchor = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体，score为带有18通道的conv1*1卷积，
        #-----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整，loc带有36通道的conv1*1卷积
        #-----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    #输入的x为feature map共享特征p2~p6层，
    def forward(self, x, img_size, scale=1.):
        rois = []       
        roi_indices =[]        
        rpn_locs =[]
        rpn_scores=[]  
        anchor = []
        #对p2~p5层分别进行建议框生成
        for p in x:
            n, _, h, w = p.shape
            #-----------------------------------------#
            #   先进行一个3x3的卷积，可理解为特征整合
            #-----------------------------------------#
            p = F.relu(self.conv1(p))  # 激活函数
            #-----------------------------------------#
            #   回归预测对先验框进行调整
            #   view(n, -1, 4)：n个（m/（4*n））行4列的新tensor形状。
            #   交换后n（第0维度）=batch_size(这里为2，表示背景和物体形状)，
            #   -1（1维度）：表示每个先验框（自行计算），
            #   4（第2维度）=调整先验框位置的四个参数。
            #-----------------------------------------#
            rpn_locs_k = self.loc(p)
            rpn_locs_k = rpn_locs_k.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            #-----------------------------------------#
            # torch.transpose():交换指定的两个维度的内容
            # torch.permute():一次性交换多个维度。
            # contiguous()：相当于是在permute(0, 2, 3, 1)tensor中复制一份，在用于view()中的tensor进行结构改变，而不影响前面的数据内容和结构。
            # torch.view():首先，view()函数会将Tensor所有维度拉平成一维（m），然后再根据传入的的维度信息重构出一个Tensor。
            #
            # Tensor与ndarray数组一样，
            #
            # 分类预测先验框内部是否包含物体
            #
            #-----------------------------------------#
            rpn_scores_k = self.score(p)
            rpn_scores_k = rpn_scores_k.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

            #--------------------------------------------------------------------------------------#
            #   进行softmax概率计算，每个先验框只有两个判别结果
            #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
            #--------------------------------------------------------------------------------------#
            rpn_softmax_scores = F.softmax(rpn_scores_k, dim=-1)
            rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
            rpn_fg_scores = rpn_fg_scores.view(n, -1)

            #------------------------------------------------------------------------------------------------#
            #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
            #------------------------------------------------------------------------------------------------#
            anchor_k = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
            rois_k = []
            roi_indices_k = []
            #分离开背景和前景 
            for i in range(n):
                roi = self.proposal_layer(rpn_locs_k[i], rpn_fg_scores[i], anchor_k, img_size, scale=scale)
                batch_index = i * torch.ones((len(roi),))
                rois_k.append(roi.unsqueeze(0))
                roi_indices_k.append(batch_index.unsqueeze(0))

            #------------------------------------------------------------------#
            #   获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
            #------------------------------------------------------------------#
            rois.append(torch.cat(rois_k, dim=0).type_as(p))
            roi_indices.append(torch.cat(roi_indices_k, dim=0).type_as(p))
            anchor.append(torch.from_numpy(anchor_k).unsqueeze(0).float().to(p.device))
            rpn_locs.append(rpn_locs_k)
            rpn_scores.append(rpn_scores_k)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels     = 512, 
        mid_channels    = 512, 
        ratios          = [0.5, 1, 2],
        anchor_scales   = [4, 16, 32], 
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        #每个网格上默认的先验框数量
        n_anchor            = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体，score为带有18通道的conv1*1卷积，
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整，loc带有36通道的conv1*1卷积
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    #输入的x为feature map共享特征层，
    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape 
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))  #激活函数
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        # view(n, -1, 4)：n个（m/（4*n））行4列的新tensor形状。
        # 交换后n（第0维度）=batch_size(这里为2，表示背景和物体形状)，
        # -1（1维度）：表示每个先验框（自行计算），
        # 4（第2维度）=调整先验框位置的四个参数。
        #-----------------------------------------#
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #-----------------------------------------#
        # torch.transpose():交换指定的两个维度的内容
        # torch.permute():一次性交换多个维度。
        # contiguous()：相当于是在permute(0, 2, 3, 1)tensor中复制一份，在用于view()中的tensor进行结构改变，而不影响前面的数据内容和结构。
        # torch.view():首先，view()函数会将Tensor所有维度拉平成一维（m），然后再根据传入的的维度信息重构出一个Tensor。
        # 
        # Tensor与ndarray数组一样，
        # 
        # 分类预测先验框内部是否包含物体
        #
        #-----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)

        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois        = list()
        roi_indices = list()
        for i in range(n):
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
