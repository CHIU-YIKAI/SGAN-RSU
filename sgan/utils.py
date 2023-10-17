import os
import time
import torch
import math
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def get_end_block_rel(curr_pos,scene_info,block_flag_num):
    batch = curr_pos.size(0)

    curr_block_rel = torch.zeros(batch, 2,block_flag_num ).cuda()
    for idx_batch, _batch in enumerate(curr_block_rel):
        for idx_xy, _xy in enumerate(_batch):
            for idx_flog, flog in enumerate(_xy):
                curr_block_rel[idx_batch][idx_xy][idx_flog] = curr_pos[idx_batch][idx_xy] - scene_info[idx_xy][idx_flog]
    #* curr_block_rel: torch.Size([201, 2, 12])
    return curr_block_rel

def tmp_get_direction(curr_pos_rel_seq, seq_len):
    direction = np.zeros(2, seq_len)
    for i in range(1, seq_len):
        if curr_pos_rel_seq[0, i]!=0 or curr_pos_rel_seq[1, i]!=0:
            vector_length = math.sqrt(curr_pos_rel_seq[0, i] * curr_pos_rel_seq[0, i] + curr_pos_rel_seq[1, i] * curr_pos_rel_seq[1, i])
            direction[0, i] = curr_pos_rel_seq[0, i] / vector_length
            direction[1, i] = curr_pos_rel_seq[1, i] / vector_length
        else:
            direction[0, i] = direction[1, i] = 0
    return direction