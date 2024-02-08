# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None, img_fea_dim=3):
        super(cylinder_fea, self).__init__()

        #gs
        #fused_fea_dim = fea_dim + img_fea_dim
        fused_fea_dim = fea_dim
                     
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fused_fea_dim),

            nn.Linear(fused_fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        # given self.fea_compre : 16
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim


        # gs
        self.fused_fea_compre = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU()
        )



    def forward(self, pt_fea, img_fea, xy_ind):
        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
            # input [100524,9]
            # output [100524, 256]
        pooled_pt_fea = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
            # output [40238,256] ######################CHECK
        
        # gs
        # process image feature
        cat_img_fea = torch.cat(img_fea, dim=0)
        pooled_img_fea = torch_scatter.scatter_max(cat_img_fea, unq_inv, dim=0)[0]
        
        if self.fea_compre:
            processed_pooled_pt = self.fea_compression(pooled_pt_fea)
            # gs
            processed_pooled_img = self.fea_compression(pooled_img_fea)
        else:
            processed_pooled_data = pooled_pt_fea
            # OUTPUT [40238,16]
        
        # gs 
        # concat pt + img
        fused_fea = torch.cat((processed_pooled_pt, processed_pooled_img), dim=1) # [pts, 256(pt_fea) + (img_fea)]
        fused_fea = self.fused_fea_compre(fused_fea)
        
        return unq, fused_fea
