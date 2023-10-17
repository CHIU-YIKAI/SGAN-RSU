import torch
import torch.nn as nn
from sgan.utils import get_end_block_rel
from sgan.pool_block_net import PoolBlockNet
from sgan.gravity import GravityNet

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

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True,
        block_flag_num=0,
        traj_frequency_map=None, args=None
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.pooling_net_num = 0

        self.pool_net_on = args.g_pool_net_on
        self.block_net_on = args.g_block_net_on
        self.block_flag_num = block_flag_num
        self.direction_on = args.g_direction_on
        self.gravity_on = args.g_gravity_on
        self.trajmap_on = args.g_trajmap_on        
        self.typeID_on = args.g_typeID_on        

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        
        if pool_every_timestep:
            if self.pool_net_on:
                self.pooling_net_num+=1
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=self.h_dim,
                    bottleneck_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm
                )
            
            if self.block_net_on:
                self.pooling_net_num+=1
                self.block_pooling_net = PoolBlockNet(
                    embedding_dim=embedding_dim,
                    h_dim=self.h_dim,
                    block_flag_num =self.block_flag_num,
                    bottleneck_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm                
                )
                
            if self.gravity_on:
                self.pooling_net_num+=1
                self.gravity_net = GravityNet(
                    embedding_dim=embedding_dim,
                    h_dim=self.h_dim,
                    block_flag_num=self.block_flag_num,
                    bottleneck_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm
                )

            if self.direction_on:
                self.pooling_net_num+=1
                mlp_direction_context_dims = [self.h_dim*2, 512, self.h_dim]
                self.mlp_direction_context = make_mlp(
                    mlp_direction_context_dims,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
                self.direction_embedding = nn.Linear(1, self.h_dim)
            
            if self.trajmap_on:
                self.pooling_net_num+=1
                self.traj_frequency_map = traj_frequency_map
                self.neighborhood_frequency_embedding=nn.Linear(9, self.h_dim)
                self.neighborhood_frequency_context= make_mlp([2*self.h_dim, 512, self.h_dim],
                activation=activation, batch_norm=batch_norm, dropout=dropout)

            if self.typeID_on:
                self.pooling_net_num+=1
                self.typeID_embedding=nn.Linear(1, self.h_dim)
                self.typeID_context = make_mlp(
                    [self.h_dim , 512, self.h_dim],
                    activation=activation, 
                    batch_norm=batch_norm, 
                    dropout=dropout
                )

            
            mlp_dims = [h_dim*(self.pooling_net_num+1), mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, direction, typeID_seq, traj_leng, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        if self.trajmap_on and self.pool_every_timestep:
            self.neighborhood_frequency = self.traj_frequency_map(last_pos, traj_leng)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                
                decoder_h_input = state_tuple[0]
                decoder_h_input = decoder_h_input[-1,:,:]
                decoder_h = decoder_h_input.view(-1, self.h_dim)
                if self.pool_net_on:                    
                    pool_h = self.pool_net(decoder_h_input, seq_start_end, curr_pos)
                    decoder_h = torch.cat(
                        [decoder_h, pool_h], dim=1)                    
                
                if self.block_net_on:
                    curr_block_rel = get_end_block_rel(curr_pos, self.scene_info, self.block_flag_num)
                    block_h = self.block_pooling_net(decoder_h_input, seq_start_end, curr_block_rel)
                    decoder_h = torch.cat(
                        [decoder_h, block_h], dim=1)
                
                if self.gravity_on:
                    curr_block_rel = get_end_block_rel(curr_pos, self.scene_info, self.block_flag_num)
                    gravity_h = self.gravity_net(decoder_h_input, seq_start_end, curr_block_rel)
                    decoder_h = torch.cat(
                        [decoder_h, gravity_h], dim=1)
                
                if self.direction_on:
                    curr_direction = direction[-1,:]
                    curr_direction = torch.unsqueeze(curr_direction,1)                
                    curr_direction = self.direction_embedding(curr_direction.type(torch.float32))
                    curr_direction_cat = torch.cat([decoder_h_input, curr_direction], dim=1)
                    direction_h = self.mlp_direction_context(curr_direction_cat)
                    decoder_h = torch.cat(
                        [decoder_h, direction_h], dim=1)

                if self.trajmap_on: 
                    trajmap_em = self.neighborhood_frequency_embedding(self.neighborhood_frequency)
                    trajmap_h = self.neighborhood_frequency_context(torch.cat([decoder_h_input.view(-1, self.h_dim), trajmap_em], dim=1))
                    decoder_h = torch.cat(
                        [decoder_h, trajmap_h], dim=1)
                
                if self.typeID_on:
                    typeID_unemb = typeID_seq[-1,:]
                    typeID_em = self.typeID_embedding(typeID_unemb)
                    typeID_h = self.typeID_context(typeID_em)
                    decoder_h = torch.cat(
                        [decoder_h, typeID_h], dim=1)

                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

                    
            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped',
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True,
        traj_frequency_map=None, traj_leng=None, args=None, block_flag_num=0
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.block_flag_num = block_flag_num
        self.direction_on = args.g_direction_on
        self.trajmap_on = args.g_trajmap_on
        self.gravity_on = args.g_gravity_on
        self.pool_net_on = args.g_pool_net_on
        self.block_net_on = args.g_block_net_on
        self.typeID_on = args.g_typeID_on
        self.pooling_net_num = 0
        self.h_dim = encoder_h_dim
        self.traj_leng = traj_leng

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            traj_frequency_map=traj_frequency_map,
            args=args
        )

        if self.pool_net_on:
            self.pooling_net_num+=1
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=self.h_dim,
                bottleneck_dim=self.h_dim,
                activation=activation,
                batch_norm=batch_norm
                )
            
        if self.block_net_on:
            self.pooling_net_num+=1
            self.block_pooling_net = PoolBlockNet(
                embedding_dim=embedding_dim,
                h_dim=self.h_dim,
                block_flag_num =self.block_flag_num,
                bottleneck_dim=self.h_dim,
                activation=activation,
                batch_norm=batch_norm                
            )
            
        if self.gravity_on:
            self.pooling_net_num+=1
            self.gravity_net = GravityNet(
                embedding_dim=embedding_dim,
                h_dim=self.h_dim,
                block_flag_num=self.block_flag_num,
                bottleneck_dim=self.h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.direction_on:
            self.pooling_net_num+=1
            mlp_direction_context_dims = [self.h_dim*2, 512, self.h_dim]
            self.mlp_direction_context = make_mlp(
                mlp_direction_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
            self.direction_embedding = nn.Linear(1, self.h_dim)
        
        if self.trajmap_on:
            self.pooling_net_num+=1
            self.traj_frequency_map=traj_frequency_map
            self.neighborhood_frequency_embedding=nn.Linear(9, self.h_dim)
            self.neighborhood_frequency_context= make_mlp(
                [self.h_dim*2, 512, self.h_dim],
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )

        if self.typeID_on:
            self.pooling_net_num+=1
            self.typeID_embedding=nn.Linear(1, self.h_dim)
            self.typeID_context = make_mlp(
                [self.h_dim , 512, self.h_dim],
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )

        #* LSTM 取代 class encoder
        self.lstm_encoder = nn.LSTM(self.h_dim, self.h_dim, num_layers, dropout=dropout)
        #* 給 LSTM 用的
        self.feature_hidden_embedding = nn.Linear(self.h_dim * (self.pooling_net_num+1), self.h_dim)
        #* 給 obj_rel 用的
        self.traj_rel_embedding = nn.Linear(2, self.h_dim)
        #* 給 lstm state h 用的
        self.lstm_state_h_embedding = nn.Linear(self.h_dim * 2, self.h_dim)  

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        input_dim = encoder_h_dim * ( 1 + self.pooling_net_num )
        
        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or #self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def init_hidden(self, num_layers, batch, h_dim):
        return (torch.zeros(num_layers, batch, h_dim).cuda(), torch.zeros(num_layers, batch, h_dim).cuda())

    def forward(self, obs_traj, obs_traj_rel, direction, typeID_seq, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        traj_rel_final_h = self.traj_rel_embedding(obs_traj_rel)
        lstm_state = self.init_hidden(self.num_layers, batch, self.h_dim)
        
        if self.trajmap_on:
            self.neighborhood_frequency = self.traj_frequency_map(obs_traj, self.traj_leng)
            
        for indx in range(self.obs_len):
            #for tensor concating all features
            features=[traj_rel_final_h[indx]]

            #* pooling net 
            if self.pool_net_on:
                classifier_pooling = self.pool_net(traj_rel_final_h[indx], seq_start_end, obs_traj[indx])
                features.append(classifier_pooling)

            #* block net
            if self.block_net_on:            
                curr_block_rel = get_end_block_rel(obs_traj[indx], self.scene_info, self.block_flag_num)
                classifier_block = self.block_pooling_net(traj_rel_final_h[indx], seq_start_end, curr_block_rel)
                features.append(classifier_block)

            #* gravity
            if self.gravity_on:
                curr_block_rel = get_end_block_rel(obs_traj[indx], self.scene_info, self.block_flag_num)
                classifier_repul_force = self.gravity_net(traj_rel_final_h[indx], seq_start_end, curr_block_rel)
                features.append(classifier_repul_force)

            #* direction attention
            if self.direction_on:
                curr_direction = direction[indx]
                curr_direction = torch.unsqueeze(curr_direction,1)
                curr_direction = self.direction_embedding(curr_direction.type(torch.float32))
                curr_direction_cat = torch.cat([traj_rel_final_h[indx],curr_direction],dim=1)  
                classifier_direction = self.mlp_direction_context(curr_direction_cat)
                features.append(classifier_direction)

            # neighborhood feature (from map)
            if self.trajmap_on:
                curr_neighborhood_frequency = self.neighborhood_frequency[indx]
                curr_neighborhood_frequency_e = self.neighborhood_frequency_embedding(curr_neighborhood_frequency)
                classifier_neighborhood_frequency = self.neighborhood_frequency_context(torch.cat([traj_rel_final_h[indx], curr_neighborhood_frequency_e],dim=1))
                features.append(classifier_neighborhood_frequency)
            
            # type ID feature
            if self.typeID_on:
                curr_typeID = typeID_seq[indx]
                curr_typeID = self.typeID_embedding(curr_typeID.type(torch.float32))
                classifier_typeID = self.typeID_context(curr_typeID)
                features.append(classifier_typeID)

            classifier_input = torch.cat(features, dim=1)
            lstm_hidden = self.feature_hidden_embedding(classifier_input)
            lstm_hidden = torch.unsqueeze(lstm_hidden, 0)


            lstm_hidden_h = lstm_state[0]
            lstm_hidden_h = torch.cat([lstm_hidden_h, lstm_hidden], dim=2)
            lstm_hidden_h = self.lstm_state_h_embedding(lstm_hidden_h)
            lstm_hidden_c = lstm_state[1]
            lstm_state = (lstm_hidden_h, lstm_hidden_c)
            lstm_input = torch.unsqueeze(traj_rel_final_h[indx],0)
            output, lstm_state = self.lstm_encoder(lstm_input, lstm_state)

        # Encode seq

        final_encoder_h = self.encoder(obs_traj_rel)
        #final_encoder_h = torch.cat([final_encoder_h, lstm_state[0]])
        
        mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Pool States
        if self.pooling_net_num == 0:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
        else:
            tmp_final_encoder_h = final_encoder_h.view(
                -1, self.encoder_h_dim)
            end_pos = obs_traj[-1, :, :]

            if self.pool_net_on:
                pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
                # Construct input hidden states for decoder
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, pool_h], dim=1)

            if self.block_net_on:
                block_h = self.block_pooling_net(end_pos, self.scene_info, self.block_flag_num)
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, block_h], dim=1)

            if self.direction_on:
                curr_direction = direction[-1,:]
                curr_direction = torch.unsqueeze(curr_direction,1)
                curr_direction = self.direction_embedding(curr_direction.type(torch.float32))
                curr_direction_cat = torch.cat([mlp_decoder_context_input, curr_direction], dim=1)
                direction_h = self.mlp_direction_context(curr_direction_cat)
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, direction_h], dim=1)

            if self.gravity_on:
                gravity_h = self.gravity_net(end_pos, self.scene_info, self.block_flag_num)
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, gravity_h], dim=1)

            if self.trajmap_on:
                trajmap_em = self.neighborhood_frequency_embedding(self.neighborhood_frequency[-1,:])
                trajmap_h = self.neighborhood_frequency_context(torch.cat([mlp_decoder_context_input, trajmap_em], dim=1))
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, trajmap_h], dim=1)

            if self.typeID_on:
                typeID_unemb = typeID_seq[-1,:,:]
                typeID_em = self.typeID_embedding(typeID_unemb)
                typeID_h = self.typeID_context(typeID_em)
                tmp_final_encoder_h = torch.cat(
                    [tmp_final_encoder_h, typeID_h], dim=1)
            
            mlp_decoder_context_input = tmp_final_encoder_h

            
        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            direction, 
            typeID_seq,
            self.traj_leng,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        block_flag_num=12, scene_info=None, num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local', traj_frequency_map=None, traj_leng=None, scene_frequency_map=None, args=None
    ):
        super(TrajectoryDiscriminator, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.traj_leng=traj_leng
        self.block_flag_num = block_flag_num
        self.scene_info = scene_info
        self.num_layers = num_layers
        self.direction_on = args.d_direction_on
        self.trajmap_on = args.d_trajmap_on
        self.gravity_on = args.d_gravity_on
        self.pool_net_on = args.d_pool_net_on
        self.block_net_on = args.d_block_net_on
        self.typeID_on = args.d_typeID_on
        self.scene_on = args.d_scene_on
        self.pooling_net_num = 0
        
        self.scene_object_types = args.scene_object_types

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        if self.pool_net_on:
            self.pooling_net_num += 1
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )
            
        if self.block_net_on:
            self.pooling_net_num += 1
            self.block_pooling_net = PoolBlockNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                block_flag_num =self.block_flag_num,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm                
            )
            
        if self.gravity_on:
            self.pooling_net_num += 1
            self.gravity_net = GravityNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                block_flag_num=self.block_flag_num,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )
            
        if self.direction_on:
            self.pooling_net_num += 1
            mlp_direction_context_dims = [h_dim*2, 512, h_dim]
            self.mlp_direction_context = make_mlp(
                mlp_direction_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
            self.direction_embedding = nn.Linear(1, h_dim)
            
        if self.trajmap_on:
            self.pooling_net_num += 1
            self.traj_frequency_map = traj_frequency_map
            self.neighborhood_frequency_embedding = nn.Linear(9,h_dim)
            self.neighborhood_frequency_context = make_mlp( 
                [h_dim * 2 , 512, h_dim],  
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )

        if self.typeID_on:
            self.pooling_net_num += 1
            self.typeID_embedding = nn.Linear(1, h_dim)
            self.typeID_context = make_mlp(
                [h_dim , 512, h_dim],
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )
        
        if self.scene_on:
            self.pooling_net_num += 1
            self.scene_frequency_map = scene_frequency_map
            sceneseg_context_dims = [h_dim, 512, h_dim]
            self.sceneseg_embedding = nn.Linear(3*self.scene_object_types, h_dim)
            # self.sceneseg_embedding = nn.Linear(2*self.scene_object_types, h_dim)
            # self.sceneseg_embedding = nn.Linear(2, h_dim)
            self.sceneseg_context = make_mlp(
                sceneseg_context_dims,
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )

        real_classifier_dims = [(self.pooling_net_num+1)*h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        
        self.traj_rel_embedding = nn.Linear(2, h_dim)

    def forward(self, traj, traj_rel, direction, typeID_seq, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        
        seq_len = traj_rel.size()[0]
        #batch = traj_rel.size()[1]
        
        traj_rel_final_h = self.traj_rel_embedding(traj_rel)
        
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.trajmap_on:
            neighborhood_frequency = self.traj_frequency_map(traj, self.traj_leng)

        if self.scene_on:
            scene_frequency = self.scene_frequency_map(traj)

        for indx in range(seq_len):
            #for tensor concating all features
            features=[]
            
            if self.pool_net_on:
                classifier_pooling = self.pool_net(traj_rel_final_h[indx], seq_start_end, traj[indx])
                features.append(classifier_pooling)

            if self.block_net_on:
                curr_block_rel = get_end_block_rel(traj[indx], self.scene_info, self.block_flag_num)
                classifier_block = self.block_pooling_net(traj_rel_final_h[indx], seq_start_end, curr_block_rel)
                features.append(classifier_block)
                
            if self.gravity_on:
                classifier_repul_force = self.gravity_net(traj_rel_final_h[indx], seq_start_end, curr_block_rel)
                features.append(classifier_repul_force)
                
            if self.direction_on:
                curr_direction = direction[indx]
                curr_direction = torch.unsqueeze(curr_direction,1)
                curr_direction = self.direction_embedding(curr_direction.type(torch.float32))
                curr_direction_cat = torch.cat([traj_rel_final_h[indx],curr_direction],dim=1)  
                classifier_direction = self.mlp_direction_context(curr_direction_cat)
                features.append(classifier_direction)
                
            if self.trajmap_on:
                curr_neighborhood_frequency = neighborhood_frequency[indx]
                curr_neighborhood_frequency_e = self.neighborhood_frequency_embedding(curr_neighborhood_frequency)
                classifier_neighborhood_frequency = self.neighborhood_frequency_context(torch.cat([traj_rel_final_h[indx],curr_neighborhood_frequency_e],dim=1)  )
                features.append(classifier_neighborhood_frequency)
                
            if self.typeID_on:
                curr_typeID = typeID_seq[indx]
                curr_typeID = self.typeID_embedding(curr_typeID.type(torch.float32))
                classifier_typeID = self.typeID_context(curr_typeID)
                features.append(classifier_typeID)

            if self.scene_on:
                curr_scene = scene_frequency[indx]
                curr_scene = self.sceneseg_embedding(curr_scene)
                classifier_scene = self.sceneseg_context(curr_scene)
                features.append(classifier_scene)

        classifier_input = final_h.squeeze()
        if len(classifier_input.size()) == 1:
            classifier_input = classifier_input.view(1, classifier_input.size()[0])

        for i in range(len(features)):
            classifier_input = torch.cat((classifier_input, features[i]), 1)

        scores = self.real_classifier(classifier_input)
        return scores
