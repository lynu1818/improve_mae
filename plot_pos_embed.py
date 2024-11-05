import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='decoder_pos_embed')
parser.add_argument('--epoch', type=str, default='')
args = parser.parse_args()


state_dict = torch.load(args.ckpt, map_location='cpu')
if 'state_dict' in state_dict:
    pos_embed = state_dict['state_dict']['model.decoder_pos_embed']
elif 'model' in state_dict:
    pos_embed = state_dict['model']['pos_embed']



pos_embed_ch = []
cls_tkns = []
for c in range(pos_embed.shape[2]):
    tmp = pos_embed[0, :, c].detach().numpy()
    size = int(np.sqrt(tmp.shape[0]))
    if size ** 2 + 1== tmp.shape[0]:
        cls_tkn = tmp[0]
        tmp = tmp[1:]
        cls_tkns.append(cls_tkn)
    pos_embed_ch.append(tmp.reshape(size, size))

dir_path = os.path.join(args.output_dir, *Path(args.ckpt).parts[-3:-1])
os.makedirs(dir_path, exist_ok=True)

for idx, p in enumerate(pos_embed_ch):
    plt.figure()
    plt.imshow(p, cmap='hot', interpolation='nearest')
    if len(cls_tkns) > 0:
        plt.title(f'Channel {idx}, cls_token: {cls_tkns[idx]:.4f}')
    else:
        plt.title(f'Channel {idx}')
    plt.savefig(os.path.join(dir_path, f'pos_embed_{idx}-{args.epoch}.png'))
    if idx > 10:
        break