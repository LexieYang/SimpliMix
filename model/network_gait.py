import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= backbone ============
from model.backbone.DGCNN import DGCNN_fs
from model.backbone.multiview import mutiview_net
from model.backbone.Gaitset_net import Gateset_net
from model.backbone.mymodel_moreview import Mymodel
from model.backbone.mymodel_pointview import pointview
# =============================


#======== fs algorithm =========
from model.fs_module.protonet import protonet
from model.fs_module.cia import CIA 
from model.fs_module.trip import trip
from model.fs_module.pointview_trip import pointview_trip
from model.fs_module.contrastive_loss_bin import Trip_CIA
from model.fs_module.MetaOp import Class_head_MetaOpt
from model.fs_module.RelationNet import RelationNet
from model.fs_module.ssl_protonet import ssl_protonet

#===============================

class fs_network(nn.Module):
    def __init__(self,k_way,n_shot,query,backbone='mymodel',fs='cia', base_classes=10):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.query=query
        self.base_classes = base_classes

        self.s_label=torch.arange(k_way)
        self.q_label=torch.arange(k_way).repeat_interleave(query)
        
        self.backbone=self.get_backbone(backbone)
        self.fs_head=self.get_fs_head(fs)
    


    def get_backbone(self,backbone):
        if backbone=='dgcnn':
            print("DGCNN is loaded")
            return DGCNN_fs()

        elif backbone=='CNN':
            print("CNN is loaded")
            return mutiview_net()
        
        elif backbone=='gaitset':
            print("gaitset is loaded")
            return Gateset_net()
        
        
        elif backbone=='mymodel':
            print('mymodel is loaded')
            return Mymodel()
        
        elif backbone=='pointview':
            print('pointview is loaded')
            return pointview()
        
        else:
            raise ValueError('Illegal Backbone')



    
    def get_fs_head(self,fs):
        if fs=='protonet':
            print("protonet is loaded")
            return protonet(self.k,self.n,self.query)
        
        elif fs=='ssl_protonet':
            print("ssl_protonet is loaded")
            return ssl_protonet(self.k, self.n, self.query, self.base_classes)

        elif fs=='cia':
            print("CIA is loaded")
            return CIA(k_way=self.k,n_shot=self.n,query=self.query)
        
        elif fs=='trip':
            print("trip is loaded")
            return trip(k_way=self.k,n_shot=self.n,query=self.query)
    
        elif fs=='pv_trip':
            print('point view trip is loaded')
            return pointview_trip(k_way=self.k,n_shot=self.n,query=self.query)
        

        elif fs=='Trip_CIA':
            print('Trip_CIA is loaded')
            return Trip_CIA(k_way=self.k,n_shot=self.n,query=self.query)

        elif fs=='MetaOp':
            print('MetaOp is loaded')
            return Class_head_MetaOpt(way=self.k,shot=self.n,query=self.query)
        
        elif fs=='Relation':
            print('RelationNet is loaded')
            return RelationNet(k_way=self.k,n_shot=self.n,query=self.query)

        else:
            raise ValueError('Illegal fs_head')
             
     
    
    def forward(self, x, ssl_sppt_target=None, n_rotation=0):
        '''
        If backbone is the gait related network
        the embeding shape is (bin,sample_num,feat_dim), like (62,20,256)
        '''
        if n_rotation == 0:
            embeding = self.backbone(x)
            preds, loss = self.fs_head(embeding,[self.s_label,self.q_label], n_rotation=n_rotation)
            
            if torch.isnan(loss):
                save_dict={}
                save_dict['inpt']=x
                torch.save(save_dict,'nan_x')
            
            assert not torch.isnan(loss)

            return preds,loss
        else:
            # add rotation to support points, so add ssl for support
            assert ssl_sppt_target!=None
            embeding = self.backbone(x)
            support_emb, query_emb = embeding[:self.k*self.n*n_rotation], embeding[self.k*self.n*n_rotation:]
            # sppt_target = torch.arange(self.k)
            # ssl_sppt_target = torch.stack([sppt_target*n_rotation+i for i in range(n_rotation)], 1).view(-1)
            preds, loss = self.fs_head(embeding, [ssl_sppt_target, self.q_label], n_rotation=n_rotation)
           
            return preds, loss


if __name__=='__main__':
    fs_net=fs_network(k_way=5,n_shot=1,query=3,backbone='mymodel',fs='trip')
    sample_inpt=torch.randn((20,3,1024))
    pred,loss=fs_net(sample_inpt)
    a=1
    
        
        
        