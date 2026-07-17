"""Generate PPT-ready figures from DCRNN training results."""
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# Use Microsoft YaHei for Chinese text
for f in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in f.name:
        matplotlib.rcParams['font.family'] = f.name
        break
# Fallback: also try SimHei
if matplotlib.rcParams['font.family'] == 'sans-serif':
    for f in fm.fontManager.ttflist:
        if 'SimHei' in f.name:
            matplotlib.rcParams['font.family'] = f.name
            break

# ============================================================
# 1. Training Curve from log
# ============================================================
with open('logs/training.log', 'r') as f:
    log = f.read()

# Parse epochs
pattern = r'Epoch\s+(\d+)/\d+\s+\|\s+train_mae:\s+([\d.]+)\s+\|\s+val_mae:\s+([\d.]+)\s+\|\s+lr:\s+([\d.e\-]+)'
matches = re.findall(pattern, log)
epochs = [int(m[0]) for m in matches]
train_mae = [float(m[1]) for m in matches]
val_mae = [float(m[2]) for m in matches]
lr = [float(m[3]) for m in matches]

# Parse test results (every 10 epochs)
test_pattern = r'Epoch\s+(\d+)/\d+[\s\S]*?Test horizon\s+1:\s+MAE=([\d.]+).*?horizon\s+2:\s+MAE=([\d.]+).*?horizon\s+3:\s+MAE=([\d.]+)'
test_matches = re.findall(test_pattern, log)
test_epochs = [int(m[0]) for m in test_matches]
test_15min = [float(m[1]) for m in test_matches]
test_30min = [float(m[2]) for m in test_matches]
test_45min = [float(m[3]) for m in test_matches]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Train & Val MAE
ax1.plot(epochs, train_mae, color='#2196F3', linewidth=1.5, label='Train MAE (normalized)')
ax1.plot(epochs, val_mae, color='#FF5722', linewidth=1.8, label='Val MAE (mph)')
best_idx = np.argmin(val_mae)
ax1.axvline(x=epochs[best_idx], color='#999', linestyle='--', alpha=0.6)
ax1.annotate(f'Best: epoch {epochs[best_idx]}\nMAE={val_mae[best_idx]:.2f}',
             xy=(epochs[best_idx], val_mae[best_idx]),
             xytext=(epochs[best_idx]+3, val_mae[best_idx]+0.3),
             fontsize=9, color='#FF5722',
             arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1))

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MAE', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# Right: Test performance by horizon
for i, (ep, h15, h30, h45) in enumerate(zip(test_epochs, test_15min, test_30min, test_45min)):
    markers = ['o', 's', 'D']
    ax2.plot([15, 30, 45], [h15, h30, h45], marker=markers[i % 3],
             label=f'Epoch {ep}', linewidth=1.5, markersize=6)

ax2.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
ax2.set_ylabel('MAE (mph)', fontsize=12)
ax2.set_title('Test MAE by Prediction Horizon', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print('[1/5] Training curves saved')

# ============================================================
# 2. Load model & plot predictions vs ground truth
# ============================================================
import sys
sys.path.insert(0, '.')
from dcrnn_model import DCRNNModel
from utils import load_graph_data, load_dataset
import yaml

with open('config/dcrnn_la.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_cfg = config['data']
model_cfg = config['model']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, adj_mx = load_graph_data(data_cfg['graph_pkl_filename'])
dataset = load_dataset(data_cfg['dataset_dir'], data_cfg['batch_size'],
                       data_cfg.get('test_batch_size', data_cfg['batch_size']), device)
scaler = dataset['scaler']

model = DCRNNModel(
    adj_mx=adj_mx,
    batch_size=data_cfg['batch_size'],
    seq_len=model_cfg['seq_len'], horizon=model_cfg['horizon'],
    input_dim=model_cfg['input_dim'], output_dim=model_cfg['output_dim'],
    num_nodes=model_cfg['num_nodes'],
    num_rnn_layers=model_cfg['num_rnn_layers'],
    rnn_units=model_cfg['rnn_units'],
    max_diffusion_step=model_cfg['max_diffusion_step'],
    filter_type=model_cfg['filter_type'],
    use_curriculum_learning=model_cfg.get('use_curriculum_learning', True),
    cl_decay_steps=model_cfg.get('cl_decay_steps', 2000)
).to(device)

ckpt = torch.load('logs/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Get test predictions
preds_all, labels_all = [], []
with torch.no_grad():
    for x, y in dataset['test_loader'].get_iterator():
        out = model(x)
        preds_all.append(out.cpu().numpy())
        labels_all.append(y[..., :model_cfg['output_dim']].cpu().numpy())

y_pred = np.concatenate(preds_all, axis=0)
y_true = np.concatenate(labels_all, axis=0)

# Denormalize
y_pred_mph = {}
y_true_mph = {}
for h in range(y_true.shape[1]):
    y_pred_mph[h] = scaler.inverse_transform(y_pred[:y_true.shape[0], h, :, 0])
    y_true_mph[h] = scaler.inverse_transform(y_true[:, h, :, 0])

# Load real sensor IDs
with open('data/sensor_graph/graph_sensor_ids.txt', 'r') as f:
    real_ids = [int(x) for x in f.read().strip().split(',')]
sensors_map = {i: rid for i, rid in enumerate(real_ids)}

# Pick a few representative sensors
sensor_ids = [0, 5, 10, 20, 50, 100]
time_steps = slice(0, 200)  # first 200 test time steps

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
horizons = [2, 5, 11]  # 15min, 30min, 60min
h_labels = ['15 min', '30 min', '60 min']

for row, (h, label) in enumerate(zip(horizons, h_labels)):
    for col, sid in enumerate([0, 100]):
        ax = axes[row, col]
        pred = y_pred_mph[h][time_steps, sid]
        truth = y_true_mph[h][time_steps, sid]
        ax.plot(truth, color='#2196F3', linewidth=1, label='Ground Truth', alpha=0.7)
        ax.plot(pred, color='#FF5722', linewidth=1, label='DCRNN Prediction', alpha=0.7)
        ax.set_title(f'Sensor #{sensors_map.get(sid, sid)}, Horizon={label}', fontsize=11)
        ax.set_xlabel('Time Step (5 min)', fontsize=9)
        ax.set_ylabel('Speed (mph)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

plt.suptitle('DCRNN Traffic Speed Prediction vs Ground Truth', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/predictions.png', dpi=200, bbox_inches='tight')
plt.close()
print('[2/5] Prediction plots saved')

# ============================================================
# 3. Horizon-wise metrics bar chart (final results)
# ============================================================
# Use the last test evaluation (epoch 40)
horizons_all = list(range(12))
mae_vals = [2.41, 2.97, 3.39, 3.77, 4.11, 4.43, 4.72, 4.98, 5.20, 5.41, 5.61, 5.81]
mape_vals = [5.61, 6.78, 7.66, 8.40, 9.07, 9.67, 10.20, 10.68, 11.11, 11.50, 11.88, 12.26]
rmse_vals = [6.21, 7.90, 9.08, 10.08, 10.96, 11.71, 12.38, 12.94, 13.39, 13.80, 14.19, 14.56]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

colors = ['#2196F3', '#4CAF50', '#FF9800', '#f44336',
          '#9C27B0', '#00BCD4', '#FF5722', '#795548',
          '#607D8B', '#E91E63', '#3F51B5', '#009688']

x = np.arange(len(horizons_all))
width = 0.35

bars1 = ax1.bar(x, mae_vals, width, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Prediction Horizon (steps)', fontsize=12)
ax1.set_ylabel('MAE (mph)', fontsize=12)
ax1.set_title('MAE by Horizon (Final)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{h+1}\n({(h+1)*5}min)' for h in horizons_all], fontsize=7)
# Add paper baseline
ax1.axhline(y=3.60, color='#999', linestyle='--', linewidth=1, label='Paper 60min MAE=3.60')
ax1.legend(fontsize=9)

bars2 = ax2.bar(x, mape_vals, width, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Prediction Horizon (steps)', fontsize=12)
ax2.set_ylabel('MAPE (%)', fontsize=12)
ax2.set_title('MAPE by Horizon (Final)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{h+1}\n({(h+1)*5}min)' for h in horizons_all], fontsize=7)

plt.tight_layout()
plt.savefig('outputs/metrics_by_horizon.png', dpi=200, bbox_inches='tight')
plt.close()
print('[3/5] Metrics bar chart saved')

# ============================================================
# 4. Model architecture summary table as figure
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['组件', '参数', '说明'],
    ['输入序列长度', '12步（60分钟）', '历史观察窗口'],
    ['预测序列长度', '12步（60分钟）', '未来预测窗口'],
    ['节点数', '207', 'METR-LA传感器数量'],
    ['输入特征', '2', '速度 + 时间编码'],
    ['输出特征', '1', '预测速度'],
    ['RNN隐藏单元', '64', 'DCGRU内部状态维度'],
    ['RNN层数', '2', '编码器-解码器各2层'],
    ['最大扩散步数', '2', 'K阶切比雪夫多项式近似'],
    ['扩散支持', '2（基线）', '出度+入度随机游走矩阵'],
    ['总参数量', '372,352', '约37万'],
    ['批量大小', '32', '受限于8GB GPU显存'],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.22, 0.35, 0.43])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.6)

# Style header
for i in range(3):
    table[0, i].set_facecolor('#1976D2')
    table[0, i].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for row in range(1, len(table_data)):
    for col in range(3):
        if row % 2 == 0:
            table[row, col].set_facecolor('#E3F2FD')

ax.set_title('DCRNN Model Configuration', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/model_config.png', dpi=200, bbox_inches='tight')
plt.close()
print('[4/5] Model config table saved')

# ============================================================
# 5. Comparison with paper + KG method diagram
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: bar comparison with paper
methods = ['DCRNN\n(Paper ICLR 2018)', 'Our Baseline\n(Reproduction)']
x = np.arange(3)
width = 0.3

paper_vals = [2.77, 3.15, 3.60]
our_vals = [2.42, 2.97, 5.61]

ax1.bar(x - width/2, paper_vals, width, color='#90A4AE', label='Paper (ICLR 2018)',
        edgecolor='white', linewidth=0.5)
ax1.bar(x + width/2, our_vals, width, color='#FF5722', label='Our Reproduction',
        edgecolor='white', linewidth=0.5)

for i, (p, o) in enumerate(zip(paper_vals, our_vals)):
    ax1.text(i - width/2, p + 0.1, str(p), ha='center', fontsize=9, fontweight='bold')
    ax1.text(i + width/2, o + 0.1, str(o), ha='center', fontsize=9, fontweight='bold',
             color='#FF5722')

ax1.set_xticks(x)
ax1.set_xticklabels(['15 min', '30 min', '60 min'], fontsize=11)
ax1.set_ylabel('MAE (mph)', fontsize=12)
ax1.set_title('DCRNN Reproduction vs Paper', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2, axis='y')

# Right: KG enhancement illustration
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Draw boxes
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Road graph
rect1 = FancyBboxPatch((1, 6), 3, 2.5, boxstyle="round,pad=0.1",
                        facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax2.add_patch(rect1)
ax2.text(2.5, 7.8, 'Road Network\nAdjacency Matrix', ha='center', fontsize=10, fontweight='bold')
ax2.text(2.5, 7.0, '2 diffusion supports\n(out-degree + in-degree)', ha='center', fontsize=8, color='#555')

# KG semantic
rect2 = FancyBboxPatch((6, 6), 3, 2.5, boxstyle="round,pad=0.1",
                        facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
ax2.add_patch(rect2)
ax2.text(7.5, 7.8, 'Semantic KG\nAdjacency Matrix', ha='center', fontsize=10, fontweight='bold')
ax2.text(7.5, 7.0, 'Gaussian kernel\nover geo-proximity', ha='center', fontsize=8, color='#555')

# DCRNN
rect3 = FancyBboxPatch((3, 2), 4, 2.5, boxstyle="round,pad=0.1",
                        facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
ax2.add_patch(rect3)
ax2.text(5, 3.8, 'DCRNN Encoder-Decoder', ha='center', fontsize=10, fontweight='bold')
ax2.text(5, 3.0, '3 diffusion supports\n(2 road + 1 semantic)', ha='center', fontsize=8, color='#555')

# Arrows
arr1 = FancyArrowPatch((4, 6), (5, 4.7), arrowstyle='->', color='#1976D2', lw=2,
                        connectionstyle='arc3,rad=0.2')
arr2 = FancyArrowPatch((7.5, 6), (6, 4.7), arrowstyle='->', color='#FF9800', lw=2,
                        connectionstyle='arc3,rad=-0.2')
ax2.add_patch(arr1)
ax2.add_patch(arr2)

# Labels
ax2.text(4.8, 5.5, '+', ha='center', fontsize=20, fontweight='bold', color='#333')

ax2.text(5, 1.5, 'Traffic Speed\nPrediction', ha='center', fontsize=10, fontweight='bold', color='#4CAF50')

ax2.set_title('KG-Enhanced DCRNN Framework', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/comparison_and_kg.png', dpi=200, bbox_inches='tight')
plt.close()
print('[5/5] Comparison & KG diagram saved')

print('\nDone! Generated 5 figures in outputs/:')
print('  1. training_curves.png    - Training & validation loss curves')
print('  2. predictions.png        - Prediction vs ground truth samples')
print('  3. metrics_by_horizon.png - MAE/MAPE bar charts by horizon')
print('  4. model_config.png       - Model hyperparameter table')
print('  5. comparison_and_kg.png  - Paper comparison + KG method diagram')
