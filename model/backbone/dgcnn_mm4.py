import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature



class DGCNN_mm4(nn.Module):
    def __init__(self, args):
        super(DGCNN_mm4, self).__init__()
        self.args = args
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.li=nn.Sequential(nn.Linear(1024,512),
                               self.bn6,
                               nn.LeakyReLU(negative_slope=0.2),
                               nn.Dropout(0.5),
                               nn.Linear(512,256)
                               )
        # self.classifier = nn.Linear(256, num_class)
                    
        
        

    def forward(self, x, target=None, mixup=False, mixup_hidden=True, mixup_alpha=None, lam=0.4):
        batch_size = x.size(0)
        assert target != None
        if mixup_hidden:
            layer_mix = random.randint(0, 5)
            # layer_mix = 0

        else:
            layer_mix = None
        out = x
        target_a = target_b = target

        if layer_mix == 0:
            out, target_a, target_b, lam = self.mixup_data4(out, target, lam=lam)

        out = get_graph_feature(out, k=self.k)
        out = self.conv1(out)
        x1 = out.max(dim=-1, keepdim=False)[0]

        out = x1
        if layer_mix == 1:
            out, target_a, target_b, lam = self.mixup_data4(out, target, lam=lam)
        out = get_graph_feature(out, k=self.k)
        out = self.conv2(out)
        x2 = out.max(dim=-1, keepdim=False)[0]

        out = x2
        if layer_mix == 2:
            out, target_a, target_b, lam = self.mixup_data4(out, target, lam=lam)
        out = get_graph_feature(out, k=self.k)
        out = self.conv3(out)
        x3 = out.max(dim=-1, keepdim=False)[0]

        out = x3
        if layer_mix == 3:
            out, target_a, target_b, lam = self.mixup_data4(out, target, lam=lam)
        out = get_graph_feature(out, k=self.k)
        out = self.conv4(out)
        x4 = out.max(dim=-1, keepdim=False)[0]
 
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        if layer_mix == 4:
            x, target_a, target_b, lam = self.mixup_data4(x, target, lam=lam)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x1 = self.li(x)
        # x1 = self.classifier(x1)
        return x1, target_a, target_b
    
      

    def uniform_mixup(self, x1, x2, lam):
        '''
        point cloud uniform sampling: sampling lambda*npoints from x1, and
        sampling (1-lambda)*npoints from x2, then concatenate them to get
        the mixed_x
        Args: 
            x1: (batch_size, feature_dimentionality, num_points)
            x2: (batch_size, feature_dimentionality, num_points)
            lam: uniformly sampled from U[0,1]
        Returns:
            mixed_x: (batch_size, feature_dimentionality, num_points)
        '''
        device = x1.device
        bs, fd, npoints = x1.shape
        # x1 = x1.permute(0, 2, 1)
        # x2 = x2.permute(0, 2, 1)
        
        npoints_x1 = int(lam * npoints)
        npoints_x2 = npoints - npoints_x1
        
        # rand_id1 = torch.randperm(npoints).to(device)
        # rand_id2 = torch.randperm(npoints).to(device)

        new_x2 = x2[:, :, :npoints_x2]
        new_x1 = x1[:, :, :npoints_x1]
        
        mixed_x = torch.cat((new_x1, new_x2), dim=-1)
        # mixed_x = mixed_x.permute(0, 2, 1)
        
        return mixed_x
        
    def mixup_data4(self, x, y, lam):

        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        if torch.cuda.is_available():
            index = index.cuda()
        mixed_x = self.uniform_mixup(x, x[index], lam)#lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam




if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--k_way', type=int, default=5)
    args.add_argument('--n_shot', type=int, default=5)
    args.add_argument('--query', type=int, default=3)
    args = args.parse_args()
    inpt = torch.rand((args.k_way*(args.n_shot+args.query),3,1024)).cuda()
    # sppt_target = [random.randint(0, 9) for _ in range(10)]
    # sppt_target = torch.Tensor(sppt_target).int().cuda()
    sppt_target = torch.arange(args.k_way).repeat_interleave(args.n_shot)
    query_target = torch.arange(args.k_way).repeat_interleave(args.query)
    target = torch.cat((sppt_target, query_target)).cuda()
    network = DGCNN(args=args).cuda()
    out_feat = network(inpt, target=target) #out_feat shape is (10,1024)
    print(out_feat.shape)
    
