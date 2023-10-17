import argparse
import os
import torch
import time

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.TrajFreqMap import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--dataset_name', default=None, type=str)
parser.add_argument('--scale', default=12.8, type=float)

def get_generator(checkpoint, dset):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        batch_norm=args.batch_norm,
        traj_frequency_map=None, #TrajFreqMap(dset.TrajMap, args.coord_scale),
        traj_leng=0, #dset.traj_leng,
        args = args,
        block_flag_num=dset.block_flag_num)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples, scale):
    ade_outer, fde_outer = [], []
    total_traj = 0
    count = 0
    t_time = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_block_rel, pred_block_gt_rel,
            obs_direction, pred_direction_gt, non_linear_ped, loss_mask, seq_start_end) = batch
            
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                start = time.time()
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, obs_direction, typeID_seq, seq_start_end)
                end = time.time()
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                
                ade.append(displacement_error(pred_traj_fake * scale, pred_traj_gt * scale, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1] * scale, pred_traj_gt[-1] * scale, mode='raw'))
                count += (pred_traj_fake.size()[1])
                t_time += end- start
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde, count, t_time


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        
        _args = AttrDict(checkpoint['args'])
        #batch_size = _args.batch_size
        if args.dataset_name:
            path = get_dset_path(args.dataset_name, args.dset_type)
        else:
            path = get_dset_path(_args.dataset_name, args.dset_type)
        dset, loader = data_loader(_args, path)
        generator = get_generator(checkpoint, dset)
        
        ade, fde, counts, times = evaluate(_args, loader, generator, args.num_samples, args.scale)
        
        print('Path: ', path)
        if args.dataset_name:
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, Time: {:.5f} (s), Counts: {}, Avg time: {:.5f}'.format(
            args.dataset_name, _args.pred_len, ade, fde, times, counts, times/counts))
        else:
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, Time: {:.5f} (s), Counts: {}, Avg time: {:.5f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde, times, counts, times/counts))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
