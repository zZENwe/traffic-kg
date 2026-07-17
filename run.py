"""Resume DCRNN training from checkpoint or start fresh."""
import argparse
import os
import sys
import torch
import yaml

from train import train
from dcrnn_model import DCRNNModel
from utils import load_graph_data, load_dataset

# Use the train function but modify for resume
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/dcrnn_la.yaml')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
parser.add_argument('--task', type=str, default='baseline', choices=['baseline', 'kg'])
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print("=" * 50)
print(f"  DCRNN Training - {args.task}")
print("=" * 50)

if args.resume and os.path.exists(args.resume):
    ckpt = torch.load(args.resume, map_location='cpu')
    print(f"Resuming from epoch {ckpt['epoch']}, val_mae={ckpt['val_loss']:.4f}")
else:
    print("Starting fresh training")

if args.task == 'kg':
    exec(open('train_kg.py').read())
else:
    train(args)
