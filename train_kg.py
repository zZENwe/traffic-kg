"""Train DCRNN with KG-enhanced adjacency matrix (3 supports instead of 2)."""
import argparse
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import yaml

from dcrnn_model import DCRNNModel
from utils import (load_graph_data, load_dataset, build_supports,
                   masked_mae_loss, masked_mae_np, masked_mape_np, masked_rmse_np)


def evaluate(model, data_loader, scaler, output_dim, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in data_loader.get_iterator():
            out = model(x)
            preds_all.append(out.cpu().numpy())
            labels_all.append(y[..., :output_dim].cpu().numpy())

    y_pred = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(labels_all, axis=0)

    results = {}
    for h in range(y_true.shape[1]):
        pred = scaler.inverse_transform(y_pred[:y_true.shape[0], h, :, 0])
        truth = scaler.inverse_transform(y_true[:, h, :, 0])
        results[h] = {
            'mae': masked_mae_np(pred, truth),
            'mape': masked_mape_np(pred, truth),
            'rmse': masked_rmse_np(pred, truth)
        }
    return results, y_pred, y_true


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load base and KG adjacency
    _, _, adj_mx = load_graph_data(data_cfg['graph_pkl_filename'])
    with open("data/sensor_graph/kg_adj.pkl", "rb") as f:
        kg_adj = pickle.load(f)
    print(f"Graph loaded: base={adj_mx.shape}, KG={kg_adj.shape}")

    # Build combined supports: dual_random_walk on road + random_walk on KG
    road_supports = build_supports(adj_mx, model_cfg['filter_type'])  # 2 supports
    kg_support = build_supports(kg_adj, "random_walk")[0]  # 1 more support
    all_supports = road_supports + [kg_support]
    print(f"Supports: {len(all_supports)} (2 road + 1 KG)")

    dataset = load_dataset(data_cfg['dataset_dir'], data_cfg['batch_size'],
                           data_cfg.get('test_batch_size', data_cfg['batch_size']), device)
    scaler = dataset['scaler']

    # Create model with extra KG support
    model = DCRNNModel(
        adj_mx=adj_mx,  # unused if we override supports, but needed for compat
        batch_size=data_cfg['batch_size'],
        seq_len=model_cfg['seq_len'], horizon=model_cfg['horizon'],
        input_dim=model_cfg['input_dim'], output_dim=model_cfg['output_dim'],
        num_nodes=model_cfg['num_nodes'],
        num_rnn_layers=model_cfg['num_rnn_layers'],
        rnn_units=model_cfg['rnn_units'],
        max_diffusion_step=model_cfg['max_diffusion_step'],
        filter_type=model_cfg['filter_type'],
        use_curriculum_learning=model_cfg.get('use_curriculum_learning', True),
        cl_decay_steps=model_cfg.get('cl_decay_steps', 2000),
        extra_supports=all_supports  # Pass pre-built supports
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'],
                                  eps=train_cfg.get('epsilon', 1e-3))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=train_cfg['steps'], gamma=train_cfg['lr_decay_ratio'])

    best_val_loss = float('inf')
    wait = 0
    patience = train_cfg['patience']
    global_step = 0
    log_dir = 'logs_kg'
    train_cfg['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for x, y in dataset['train_loader'].get_iterator():
            global_step += 1
            out = model(x, labels=y, global_step=global_step)
            loss = masked_mae_loss(out, y[..., :model_cfg['output_dim']])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get('max_grad_norm', 5))
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)
        lr = optimizer.param_groups[0]['lr']

        val_results, _, _ = evaluate(model, dataset['val_loader'], scaler,
                                      model_cfg['output_dim'], device)
        val_loss = np.mean([v['mae'] for v in val_results.values()])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"train_mae: {train_loss:.2f} | val_mae: {val_loss:.2f} | "
              f"lr: {lr:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(log_dir, 'best_model_kg.pt'))
            print(f"  -> Saved (val_mae={val_loss:.4f})")
        else:
            wait += 1
            if wait > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    print("\n=== KG-Enhanced Final Results ===")
    checkpoint = torch.load(os.path.join(log_dir, 'best_model_kg.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results, preds, truths = evaluate(model, dataset['test_loader'], scaler,
                                            model_cfg['output_dim'], device)
    for h in [2, 5, 11]:  # 15min, 30min, 60min
        m = test_results[h]
        label = ['15min', '30min', '60min'][[2, 5, 11].index(h)]
        print(f"Horizon {label}: MAE={m['mae']:.2f}, MAPE={m['mape']:.2f}%, RMSE={m['rmse']:.2f}")

    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dcrnn_la.yaml')
    args = parser.parse_args()
    train(args)
