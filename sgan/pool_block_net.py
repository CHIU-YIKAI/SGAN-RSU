#from sgan_code.components.helpers import make_mlp
import torch.nn as nn
import torch

#* 加入障礙標記向量特徵
class PoolBlockNet(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, bottleneck_dim=1024,
        activation='relu', block_flag_num = 2,batch_norm=True, dropout=0.0
        ):
        super(PoolBlockNet,self).__init__()
        self.h_him = h_dim
        self.block_flag_num = block_flag_num
        self.embedding_dim = embedding_dim
        #* 方法 1 
        self.spatial_embedding = nn.Linear(self.block_flag_num * 2, self.block_flag_num * 16)
        #* 方法 2
        # self.spatial_embedding = nn.Linear(self.block_flag_num , self.block_flag_num * 8)
        #* 方法 1 
        mlp_pre_dim = self.block_flag_num * 16 + h_dim
        #* 方法 2
        # mlp_pre_dim = self.block_flag_num * 8 + h_dim
        
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)
    
    #* 方法 1 x,y 距離差
    def forward(self,h_state,seq_start_end,end_block_pos_rel):
        block_h = []
        #* h_state.shape = [1,852,32]
        #* seq_start_end.shape = [64,2]
        batch = end_block_pos_rel.size(0)
        for _,(start,end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            #* curr_block_pos_rel = torch.Size([17, 12, 2])
            curr_block_pos_rel = end_block_pos_rel[start:end].permute(0,2,1)
            #* curr_block_pos_rel = torch.Size([17, 24])
            curr_block_pos_rel = curr_block_pos_rel.contiguous().view(end-start, -1)
            #* curr_hidden.shape = [17,32]
            curr_hidden = h_state.view(-1, self.h_him)[start:end]
            #* curr_rel_embedding.shape = [17,16]
            curr_rel_embedding = self.spatial_embedding(curr_block_pos_rel)
            #* mlp_h_input.shape = [17,48]
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden], dim=1)
            #* curr_pool_h: torch.Size([17, 8])
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            block_h.append(curr_pool_h)
        block_h = torch.cat(block_h,dim=0)
        return block_h
        
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)