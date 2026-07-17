"""Write model predictions to Neo4j sensor nodes (memory efficient)."""
import numpy as np
import torch
import yaml
import pandas as pd
from neo4j import GraphDatabase

from dcrnn_model import DCRNNModel
from utils import load_graph_data, StandardScaler
from config import NEO4J_URI, NEO4J_AUTH

URI = NEO4J_URI
AUTH = NEO4J_AUTH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load config
with open('config/dcrnn_la.yaml', 'r') as f:
    config = yaml.safe_load(f)
data_cfg, model_cfg = config['data'], config['model']

# Load adj and compute scaler stats from train set (just stats, not full data)
_, _, adj_mx = load_graph_data(data_cfg['graph_pkl_filename'])
print(f"Adj: {adj_mx.shape}")

train_x = np.load(f"{data_cfg['dataset_dir']}/train.npz", mmap_mode='r')['x']
scaler = StandardScaler(mean=float(np.mean(train_x[..., 0])), std=float(np.std(train_x[..., 0])))
print(f"Scaler: mean={scaler.mean:.2f}, std={scaler.std:.2f}")
del train_x

# Load only test data
test = np.load(f"{data_cfg['dataset_dir']}/test.npz", mmap_mode='r')
x_test = test['x'].copy()
y_test = test['y'].copy()
test.close()

x_test[..., 0] = scaler.transform(x_test[..., 0])
y_test[..., 0] = scaler.transform(y_test[..., 0])
print(f"Test: x={x_test.shape}, y={y_test.shape}")

# Build model
model = DCRNNModel(
    adj_mx=adj_mx, batch_size=data_cfg['batch_size'],
    seq_len=model_cfg['seq_len'], horizon=model_cfg['horizon'],
    input_dim=model_cfg['input_dim'], output_dim=model_cfg['output_dim'],
    num_nodes=model_cfg['num_nodes'], num_rnn_layers=model_cfg['num_rnn_layers'],
    rnn_units=model_cfg['rnn_units'], max_diffusion_step=model_cfg['max_diffusion_step'],
    filter_type=model_cfg['filter_type'],
    use_curriculum_learning=model_cfg.get('use_curriculum_learning', True),
    cl_decay_steps=model_cfg.get('cl_decay_steps', 2000),
).to(device)

ckpt = torch.load('logs/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Model loaded (epoch {ckpt['epoch']}, val_mae={ckpt['val_loss']:.4f})")

# Run inference in batches
model.eval()
batch_size = 16  # smaller batch to save memory
num_samples = x_test.shape[0]
num_nodes = model_cfg['num_nodes']
horizon = model_cfg['horizon']

preds_all = np.zeros((num_samples, horizon, num_nodes, 1), dtype=np.float32)
with torch.no_grad():
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        x_batch = torch.FloatTensor(x_test[start:end]).to(device)
        out = model(x_batch).cpu().numpy()
        preds_all[start:end] = out
        if (start // batch_size + 1) % 50 == 0:
            print(f"  Inference {end}/{num_samples}")

print(f"Predictions: {preds_all.shape}")

# Compute per-sensor metrics
print("Computing per-sensor metrics...")
sensor_metrics = {}
for sid_idx in range(num_nodes):
    true_sid = scaler.inverse_transform(y_test[:, :, sid_idx, 0])
    pred_sid = scaler.inverse_transform(preds_all[:, :, sid_idx, 0])
    mae_h = np.mean(np.abs(pred_sid - true_sid), axis=0)
    sensor_metrics[sid_idx] = {
        'mae_15': float(mae_h[2]),
        'mae_30': float(mae_h[5]),
        'mae_60': float(mae_h[11]),
    }

all_15 = [m['mae_15'] for m in sensor_metrics.values()]
all_30 = [m['mae_30'] for m in sensor_metrics.values()]
all_60 = [m['mae_60'] for m in sensor_metrics.values()]
print(f"Overall MAE: 15min={np.mean(all_15):.2f}, 30min={np.mean(all_30):.2f}, 60min={np.mean(all_60):.2f}")
print(f"Per-sensor range: 15min [{min(all_15):.2f}, {max(all_15):.2f}], "
      f"30min [{min(all_30):.2f}, {max(all_30):.2f}], 60min [{min(all_60):.2f}, {max(all_60):.2f}]")

# Write to Neo4j
print("Writing to Neo4j...")
sensors = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
sensor_ids = [int(sensors.iloc[i]['sensor_id']) for i in range(num_nodes)]

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

with driver.session() as session:
    for idx, sid in enumerate(sensor_ids):
        m = sensor_metrics[idx]
        session.run("""
            MATCH (s:Sensor {sid: $sid})
            SET s.mae_15min = $m15,
                s.mae_30min = $m30,
                s.mae_60min = $m60
        """, sid=sid, m15=m['mae_15'], m30=m['mae_30'], m60=m['mae_60'])

    print(f"  Updated {num_nodes} sensors")

    r = session.run("""
        MATCH (s:Sensor)
        RETURN count(s) as total,
        avg(s.mae_15min) as a15, avg(s.mae_30min) as a30, avg(s.mae_60min) as a60
    """).single()
    print(f"  Verified: {r['total']} sensors, MAE 15/30/60min: {r['a15']:.2f}/{r['a30']:.2f}/{r['a60']:.2f}")

driver.close()
print("Done!")
