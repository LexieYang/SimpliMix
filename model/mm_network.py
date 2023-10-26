import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= backbone ============

from model.backbone.DGCNN import DGCNN_fs

from model.backbone.dgcnn_mm4 import DGCNN_mm4

# =============================


#======== fs algorithm =========
from model.fs_module.protonet import protonet
from model.fs_module.cia import CIA 

#===============================

class fs_network(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k_way
        self.n = cfg.n_shot
        self.query = cfg.query
        # self.base_classes = cfg.base_classes
        self.s_label=torch.arange(cfg.k_way).repeat_interleave(cfg.n_shot)
        self.q_label=torch.arange(cfg.k_way).repeat_interleave(cfg.query)
        self.label = torch.cat((self.s_label, self.q_label))
        if torch.cuda.is_available():
            self.label = self.label.cuda()
            self.q_label = self.q_label.cuda()
            self.s_label = self.s_label.cuda()
        self.backbone = self.get_backbone(cfg.backbone)
        self.fs_head = self.get_fs_head(cfg.fs_head) #protonet(self.k,self.n,self.query)
    
    def get_backbone(self, backbone):

        if backbone == 'dgcnn_mm4':
            print("dgcnn_mm4 backbone")
            return DGCNN_mm4(args=self.cfg)
        
        elif backbone == 'dgcnn':
            print("dgcnn backbone")
            return DGCNN_fs()
        
        else:
            raise Exception("backbone error")

    def get_fs_head(self, fs_head):
        if fs_head == 'cia':
            print("cia head")
            return CIA(self.k, self.n, self.query)
        elif fs_head == 'protonet':
            print("protonet head")
            return protonet(self.k,self.n,self.query)
        
        else:
            raise Exception('fs_head error')
    
    def forward(self, x, mixup_hidden=True, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        qry_lam = np.random.beta(alpha, alpha)
        sppt_lam = np.random.beta(a=10.0, b=2.0)
        if self.cfg.backbone == 'dgcnn': # the original dgcnn (no mixup)
            print("1")
            embeding = self.backbone(x)
            pred = self.fs_head(embeding)
            loss = F.cross_entropy(pred, self.q_label)
            return pred, loss
        
        else:
            embeding, target_a, target_b = self.backbone(x, target=self.label, mixup_hidden=mixup_hidden, lam=lam)
            pred = self.fs_head(embeding)
            if mixup_hidden:
                # comptue mixup ce loss
                qry_target_a, qry_target_b = target_a[self.k*self.n:], target_b[self.k*self.n:]
                loss_0, loss_1 = F.cross_entropy(pred, qry_target_a), F.cross_entropy(pred, qry_target_b)
                # loss = lam*loss_0 + (1-lam)*loss_1
                loss = loss_0 + loss_1
            else:
                loss = F.cross_entropy(pred, self.q_label)
            return pred, loss


if __name__=='__main__':
    fs_net=fs_network(k_way=5,n_shot=1,query=3,backbone='mymodel',fs='trip')
    sample_inpt=torch.randn((20,3,1024))
    pred,loss=fs_net(sample_inpt)
    a=1
    
        
        
        