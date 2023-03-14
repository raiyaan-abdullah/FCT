# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 00:54:08 2023

@author: Roy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.init as init
#from torch.autograd import Variable
#import torchvision.models as models
#import numpy as np
#import math

#from model.utils.config import cfg
#from model.rpn.rpn import _RPN
#from model.roi_layers import ROIAlign, ROIPool
#from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
#from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
#from model.framework.resnet import resnet50


class DAnARCNN(nn.Module):
    """ Dual Awareness Attention Faster R-CNN """
    def __init__(self, pool_feat_dim, q_shape, support_shape, n_way=2, n_shot=10, pos_encoding=False):
        super(DAnARCNN, self).__init__()
        #self.n_classes = classes
        self.n_way = n_way
        self.n_shot = n_shot
        #self.channel_gamma = gamma
        #self.unary_gamma = 0.1
        #self.semantic_enhance = semantic_enhance
        #self.rpn_reduce_dim = rpn_reduce_dim
        # self.model_path = "resnet50.pth"
        # self.pretrained = True
        
        # My code starts here
        # resnet = resnet50()
        # if self.pretrained == True:
        #     print("Loading pretrained weights from %s" %(self.model_path))
        #     state_dict = torch.load(self.model_path)
        #     resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
        #self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.pool_feat_dim = pool_feat_dim
        self.support_shape = support_shape
        self.q_shape = q_shape
        self.w_e = nn.Linear(self.pool_feat_dim, 1)
        
        self.w_r = nn.Linear(self.pool_feat_dim, support_shape)
        self.w_k = nn.Linear(self.pool_feat_dim, support_shape)
        self.w_q = nn.Linear(self.pool_feat_dim, q_shape)
        
        self.pos_encoding = pos_encoding
        # if pos_encoding:
        #     self.pos_encoding_layer = PositionalEncoding()
        #     self.rpn_pos_encoding_layer = PositionalEncoding(max_len=400)
        
        


    #def forward(self, query_feats, support_feats_base):
    def forward(self, support_feats_base):
        #batch_size = query_feats.size(0)

        # feature extraction
        #query_feats = self.RCNN_base(im_data)
        #support_feats_base = self.RCNN_base(support_ims)  # [B*2*shot, 1024, 20, 20]
        
        # BA Block
        support_feats_batch = support_feats_base.size(0) 
        support_feats_channel = support_feats_base.size(2) 
        #support_feats_h = support_feats_base.size(2)
        support_feats_wh = support_feats_base.size(1)
        
        #support_feats_linear = support_feats_base.view(-1, support_feats_channel, support_feats_wh)
        support_feats = self.w_e(support_feats_base)
        support_feats = F.softmax(support_feats, dim=1)
        #support_feats = support_feats.view(-1, 1, support_feats_h * support_feats_w)
        #support_feats_base_ = support_feats_base #.transpose(1, 2)
        support_feats = torch.bmm(support_feats.transpose(1, 2), support_feats_base)
        support_feats = F.leaky_relu(support_feats)
        Z = support_feats_base + support_feats
        
        return Z
        # Z[1] = support_feats_base[1] + support_feats[2]
        
        # X_feats_h = query_feats.size(2)
        # X_feats_w = query_feats.size(3)
        # X_feats = query_feats.views(-1, query_feats.size(1) * X_feats_h * X_feats_w)
        # X_feats = self.w_q(query_feats)
        
        # # CISA Block
        # Z_linear = Z.view(-1, Z.size(1) * Z.size(2) * Z.size(3))
        # Z_r = self.w_r(Z_linear)
        # Z_r = F.softmax(Z_r, dim=1)
        
        # Z_k = self.w_k(Z_linear)
        # k_q = torch.bmm(Z_k.transpose(1, 2), X_feats)
        # k_q = F.softmax(k_q, dim=1)
        
        # Z_x = Z_r + k_q
        
        # Z = Z.view(-1, Z.size(1), Z.size(2) * Z.size(3))
        # qpa = torch.bmm(Z, Z_x.transpose(1, 2))