"""Quick test to verify DCRNN model works with random data."""
import numpy as np
import torch
from dcrnn_model import DCRNNModel

adj_mx = np.random.rand(207, 207)
adj_mx = np.maximum(adj_mx, adj_mx.T)
np.fill_diagonal(adj_mx, 0)

device = torch.device('cpu')
batch_size = 64

model = DCRNNModel(
    adj_mx=adj_mx,
    batch_size=batch_size,
    seq_len=12, horizon=12, input_dim=2, output_dim=1,
    num_nodes=207, num_rnn_layers=2, rnn_units=64,
    max_diffusion_step=2, filter_type='dual_random_walk',
    use_curriculum_learning=True, cl_decay_steps=2000
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {num_params:,}')

# Training mode
x = torch.randn(batch_size, 12, 207, 2)
y = torch.randn(batch_size, 12, 207, 1)
out = model(x, labels=y, global_step=100)
print(f'Train output shape: {out.shape}')  # (64, 12, 207, 1)

# Eval mode
model.eval()
with torch.no_grad():
    out = model(x)
    print(f'Eval output shape: {out.shape}')

# One training step
model.train()
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
out = model(x, labels=y, global_step=1)
loss = loss_fn(out, y)
loss.backward()
optimizer.step()
print(f'Loss: {loss.item():.4f}')
print('All tests passed!')
