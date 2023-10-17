import torch.nn as nn
import torch

class GravityNet(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 h_dim=64,
                 bottleneck_dim=1024,
                 activation='relu',
                 block_flag_num=2,
                 batch_norm=True,
                 dropout=0.0):
        """Create a Gravity neural network.
        
        Keyword Arguments:
            embedding_dim {int} -- Embedding size. (default: {64})
            h_dim {int} -- Hidden layer size. (default: {64})
            bottleneck_dim {int} -- Bottleneck size. (default: {1024})
            activation {str} -- Activation function name. (default: {'relu'})
            block_flag_num {int} -- The number of obstacles. (default: {2})
            batch_norm {bool} -- Whether to use batch normalization or not. (default: {True})
            dropout {float} -- Dropout rate. (default: {0.0})
        """
        super(GravityNet, self).__init__()
        self.biker_mass = torch.randn((1,)).cuda()
        self.obstacle_mass = torch.randn((block_flag_num,)).cuda()
        self.g_const = 1

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.block_flag_num = block_flag_num
        self.bottleneck_dim = bottleneck_dim

        self.spatial_embedding = nn.Linear(block_flag_num * 2, block_flag_num * 16)

        mlp_pre_dim = block_flag_num * 16 + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
    
    def forward(self, 
                h_state: torch.Tensor,
                seq_start_end: torch.Tensor, 
                curr_block_rel: torch.Tensor) -> torch.Tensor:
        """Calculate gravity forces.
        
        Arguments:
            h_state {torch.Tensor} -- Hidden state from relative trajactory embedding (traj_rel_embedding).
            seq_start_end {torch.Tensor} -- Start-end slices.
            curr_block_rel {torch.Tensor} -- Different vectors between vehicles and obstacles.
        
        Returns:
            torch.Tensor -- Result.
        """
        # Distances between vehicles and vehicles
        distances = curr_block_rel.permute(0, 2, 1).norm(dim=2)

        # normalized_block_rel
        # Normalized repulsive force vectors where directions are from obstacles to vehicles
        norm_repulsive_forces = - curr_block_rel.permute(0, 2, 1) / torch.unsqueeze(distances, dim=2)

        # Calculate forces between obstacles and vehicles
        forces = self.g_const * (self.biker_mass * self.obstacle_mass / distances)

        # Repulsive force vectors where directions are from obstacles to vehicles
        repulsive_forces = norm_repulsive_forces * torch.unsqueeze(forces, dim=2)

        forces_h = []

        # batch = curr_block_rel.size(0)
        for _,(start,end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()

            repul_forces_seq = repulsive_forces[start:end]                          # [vehicles, obstacles, xy]
            repul_forces_seq = repul_forces_seq.contiguous().view(end - start, -1)  # [vehicles, obstacles * xy]

            curr_hidden = h_state.view(-1, self.h_dim)[start:end]                   # [vehicles, h_dim]

            curr_rel_embedding = self.spatial_embedding(repul_forces_seq)           # [vehicles, obstacles * 16]

            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden], dim=1)       # [vehicles, obstacles * 16 + h_dim]

            curr_pool_h = self.mlp_pre_pool(mlp_h_input)                            # [vehicles, bottleneck_dim]

            forces_h.append(curr_pool_h)

        forces_h = torch.cat(forces_h, dim=0)   # [batch_size (vehicles), bottleneck_dim]
        return forces_h
        
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