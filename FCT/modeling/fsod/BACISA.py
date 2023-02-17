import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math


from model.utils.config import cfg
#from model.rpn.rpn import _RPN
#from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.framework.resnet import resnet50

class DAnARCNN(nn.Module):
    """ Dual Awareness Attention Faster R-CNN """
    def __init__(self, classes, rpn_reduce_dim, gamma, semantic_enhance, n_way=2, n_shot=10, pos_encoding=True):
        super(DAnARCNN, self).__init__()
        self.n_classes = classes
        self.n_way = n_way
        self.n_shot = n_shot
        self.channel_gamma = gamma
        self.unary_gamma = 0.1
        self.semantic_enhance = semantic_enhance
        self.rpn_reduce_dim = rpn_reduce_dim
        self.model_path = "resnet50.pth"
        self.pretrained = True
        
        # few shot rcnn head
        self.pool_feat_dim = 1024
        self.rcnn_dim = 64
        self.avgpool = nn.AvgPool2d(14, stride=1)
        dim_in = self.pool_feat_dim
        ################
        self.rpn_unary_layer = nn.Linear(dim_in, 1)
        init.normal_(self.rpn_unary_layer.weight, std=0.01)
        init.constant_(self.rpn_unary_layer.bias, 0)
        self.rcnn_unary_layer = nn.Linear(dim_in, 1)
        init.normal_(self.rcnn_unary_layer.weight, std=0.01)
        init.constant_(self.rcnn_unary_layer.bias, 0)
        
        self.rpn_adapt_q_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q_layer.bias, 0)
        self.rpn_adapt_k_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k_layer.bias, 0)
        self.pos_encoding = pos_encoding
        
        if pos_encoding:
            self.pos_encoding_layer = PositionalEncoding()
            self.rpn_pos_encoding_layer = PositionalEncoding(max_len=200)
        
        resnet = resnet50()
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet. (base)
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        
    def forward(self, im_data, support_ims, all_cls_gt_boxes=None):

        batch_size_x = im_data.size(0)
        batch_size_y = support_ims.size(0)

        # feature extraction
        base_feat = self.RCNN_base(im_data)
        if self.training:
            support_ims = support_ims.view(-1, support_ims.size(1), support_ims.size(2), support_ims.size(3))
            support_feats = self.RCNN_base(support_ims)  # [B*2*shot, 1024, 20, 20]
            support_feats = support_feats.view(-1, self.n_way*self.n_shot, support_feats.size(0), support_feats.size(1), support_feats.size(2))
        #    support_feats = support_feats.view(support_feats.size(0), self.n_way*self.n_shot, support_feats.size(1), -1, support_feats.size(2))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].contiguous()  # [B, shot, 1024, 20, 20]
            #neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].contiguous()
            # pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            # neg_support_feat_pooled = self.avgpool(neg_support_feat.view(-1, 1024, 20, 20))
            # pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, -1, 7)  # [B, shot, 1024, 7, 7]
            # neg_support_feat_pooled = neg_support_feat_pooled.view(batch_size, self.n_shot, 1024, -1, 7)
        else:
            support_ims = support_ims.view(-1, support_ims.size(1),  support_ims.size(2),  support_ims.size(3))
            support_feats = self.RCNN_base(support_ims)
            #support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(0), support_feats.size(1), support_feats.size(2))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :]
            # pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            # pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)

        batch_size = pos_support_feat.size(0)
        feat_h = base_feat.size(2)
        feat_w = base_feat.size(3)
        support_mat = pos_support_feat.transpose(0, 1).view(self.n_shot, batch_size, 1024, -1).transpose(2, 3)  # [shot, B, 400, 1024]
        query_mat = base_feat.view(batch_size_x, 1024, -1).transpose(1, 2)  # [B, h*w, 512]

        dense_support_feature = []
        dense_query_feature = []
        q_matrix = self.rpn_adapt_q_layer(query_mat)  # [B, hw, 256]
        q_matrix = q_matrix - q_matrix.mean(1, keepdim=True)
        for i in range(self.n_shot):
            if self.pos_encoding:
                single_s_mat = self.rpn_pos_encoding_layer(support_mat[i], self.training)  # [B, 400, 1024]
            else:
                single_s_mat = self.support_mat[i]

            # support channel enhance (BA Block)
            if self.semantic_enhance:
                support_spatial_weight = self.rpn_channel_k_layer(single_s_mat)  # [B, 400, 1]
                support_spatial_weight = F.softmax(support_spatial_weight, 1)
                support_channel_global = torch.bmm(support_spatial_weight.transpose(1, 2), single_s_mat)  # [B, 1, 512]
                single_s_mat = single_s_mat + self.channel_gamma * F.leaky_relu(support_channel_global) # Value of Z in the paper

            # support adaptive attention
            k_matrix = self.rpn_adapt_k_layer(single_s_mat)  # [B, 400, 256]
            k_matrix = k_matrix - k_matrix.mean(1, keepdim=True)
            if not self.training:
                k_matrix = k_matrix.view(batch_size_x, -1, 256)
                single_s_mat = single_s_mat.reshape(1, -1, 1024)
                
            support_adaptive_attention_weight = torch.bmm(q_matrix, k_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim)
            #support_adaptive_attention_weight = torch.bmm(q_matrix, k_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim)  # [B, hw, 400]
            support_adaptive_attention_weight = F.softmax(support_adaptive_attention_weight, dim=2) # Attention maps
            unary_term = self.rpn_unary_layer(single_s_mat)  # [B, 400, 1]
            query_adaptive_attention_feature = F.softmax(unary_term, dim=1).transpose(1, 2)
            unary_term = unary_term.transpose(1, 2)
            #support_adaptive_attention_weight = support_adaptive_attention_weight + self.unary_gamma * unary_term.transpose(1, 2)  # [B, hw, 400]
            
            support_adaptive_attention_feature = torch.bmm(support_adaptive_attention_weight, single_s_mat)  # [B, hw, 1024]
            query_adaptive_attention_feature = torch.bmm(unary_term, single_s_mat)  # [B, hw, 1024]

            dense_support_feature += [support_adaptive_attention_feature]
            dense_query_feature += [query_adaptive_attention_feature]
        #dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)
        dense_support_feature = torch.stack(dense_support_feature, 0)# [B, hw, 1024]
        dense_query_feature = torch.stack(dense_query_feature, 0).mean(0)  # [B, hw, 1024]
        
        # dense_support_feature = dense_support_feature.transpose(1, 2).view(batch_size_y, 1024, feat_h, feat_w)
        # dense_query_feature = dense_query_feature.transpose(1, 2).view(batch_size_x, 1024, 1, 1)
        if self.training:
            dense_support_feature = dense_support_feature.transpose(1, 2).view(batch_size_y, 1024, feat_h, feat_w)
            dense_query_feature = dense_query_feature.transpose(1, 2).view(batch_size_x, 1024, 1, 1)
        else:
            dense_support_feature = dense_support_feature.transpose(1, 2).reshape(batch_size_y, 1024, -1, 1)
            dense_query_feature = dense_query_feature.transpose(1, 2).reshape(batch_size_x, 1024, -1, 1)
        
        return dense_query_feature, dense_support_feature

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=1024, max_len=200):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False)
        
        pe1 = torch.zeros(20, d_model)
        position1 = torch.arange(0., 20).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe1[:, 0::2] = torch.sin(position1 * div_term1)
        pe1[:, 1::2] = torch.cos(position1 * div_term1)
        self.pe1 = Variable(pe1.unsqueeze(0), requires_grad=False)

    def forward(self, x, train):
        # print(f"{x.size()}, {self.pe.to(x.device).size()}")
        if train:
            x = x + self.pe.to(x.device)
        else:
            x = x + self.pe1.to(x.device)
        return x
