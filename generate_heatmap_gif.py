"""Generate animated heatmap GIF from METR-LA traffic data.

Output: outputs/traffic_heatmap.gif — color-coded sensors over time.
Green=fast, Yellow=medium, Red=congested. Node size ~ congestion.
"""
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import os
from datetime import datetime, timezone

# ---- Config ----
OUTPUT = 'outputs/traffic_heatmap.gif'
FPS = 8
SAMPLE_STEP = 12          # 12 x 5min = 1 hour per frame
MAX_FRAMES = 72           # 72 hours = 3 days for a ~9s GIF at 8fps
DPI = 150
FIG_W, FIG_H = 16, 7      # inches (wide aspect)

# ---- Load data ----
base = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(base, 'data', 'metr-la.h5')
csv_path = os.path.join(base, 'data', 'sensor_graph', 'graph_sensor_locations.csv')
os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)

print("Loading METR-LA data...")
with h5py.File(h5_path, 'r') as f:
    vals = f['data/block0_values'][:]     # (34272, 207)
    times_raw = f['data/axis1'][:]

times_ns = times_raw.astype(np.int64)
if times_ns[0] > 1e15:
    times_s = times_ns / 1e9
else:
    times_s = times_ns

# Downsample
vals_ds = vals[::SAMPLE_STEP, :][:MAX_FRAMES]
times_ds = times_s[::SAMPLE_STEP][:MAX_FRAMES]
timestamps = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%m/%d %H:%M') for t in times_ds]
n_frames, n_sensors = vals_ds.shape
print(f"  {n_frames} frames x {n_sensors} sensors")

# Load sensor coords
sensors = pd.read_csv(csv_path, index_col=0)
lats = sensors['latitude'].values[:n_sensors]
lons = sensors['longitude'].values[:n_sensors]

# Normalize lon/lat to canvas coords (preserve aspect ratio)
lon_min, lon_max = lons.min(), lons.max()
lat_min, lat_max = lats.min(), lats.max()
lon_range = lon_max - lon_min
lat_range = lat_max - lat_min
aspect = lon_range / lat_range

x = (lons - lon_min) / lon_range
y = (lat_max - lats) / lat_range  # flip so north is up

# Rotate 180: flip y, then stretch x 40% wider
y = 1 - y
x = (x - 0.5) * 1.4 + 0.5

# ---- Build edge lines for background ----
print("Building road network overlay...")
coords = sensors[['latitude', 'longitude']].values
geo_dist = np.sqrt((coords[:, None, 0] - coords[None, :, 0]) ** 2 +
                   (coords[:, None, 1] - coords[None, :, 1]) ** 2)
edge_segments = []
for i in range(n_sensors):
    nearest = np.argsort(geo_dist[i])[1:6]
    for j in nearest:
        edge_segments.append([(x[i], y[i]), (x[j], y[j])])
print(f"  {len(edge_segments)} edges")

# ---- Render each frame ----
print(f"Rendering {n_frames} frames...")
frames = []
vmin, vmax = 0, 70

for frame_idx in range(n_frames):
    speeds = vals_ds[frame_idx, :]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    # Road network edges
    lc = LineCollection(edge_segments, colors='#0f3460', linewidths=0.3, alpha=0.5)
    ax.add_collection(lc)

    # Sensor nodes: size by congestion (bigger = slower), color by speed
    sizes = np.where(speeds > 0, 25 + (1 - np.clip(speeds / vmax, 0, 1)) * 55, 8)
    colors = np.zeros((n_sensors, 4))
    for i in range(n_sensors):
        spd = speeds[i]
        if spd <= 0:
            colors[i] = [0.3, 0.3, 0.3, 0.3]
        else:
            t = np.clip(spd / vmax, 0, 1)
            if t < 0.5:
                s = t / 0.5
                colors[i] = [1.0, 0.3 + s * 0.6, 0.3, 0.85]  # red→yellow
            else:
                s = (t - 0.5) / 0.5
                colors[i] = [1.0 - s * 0.85, 0.9 - s * 0.1, 0.3, 0.85]  # yellow→green

    ax.scatter(x, y, s=sizes, c=colors, edgecolors='white', linewidths=0.15, zorder=5)

    # Title & timestamp
    ax.text(0.5, 1.02, 'LA Highway Traffic — Spatio-Temporal Heatmap',
            transform=ax.transAxes, ha='center', fontsize=14, color='#e94560',
            fontweight='bold', fontfamily='sans-serif')
    ax.text(0.5, 0.97, f'METR-LA · 207 Sensors · {timestamps[frame_idx]}',
            transform=ax.transAxes, ha='center', fontsize=10, color='#888')

    # Legend
    legend_y = 0.05
    for label, c, sx, sy in [
        ('Congested <40 mph', '#ef4444', 0.18, legend_y),
        ('Medium 40-60 mph', '#eab308', 0.40, legend_y),
        ('Free Flow >=60 mph', '#22c55e', 0.63, legend_y),
    ]:
        ax.text(sx, sy, label, transform=ax.transAxes, fontsize=8, color=c, ha='center')

    # Gradient bar (drawn as horizontal colored rectangles at bottom)
    bar_y_bottom, bar_height = 0.01, 0.025
    bar_steps = 80
    for k in range(bar_steps):
        bx0 = 0.15 + k / bar_steps * 0.70
        bx1 = 0.15 + (k + 1) / bar_steps * 0.70
        t_val = k / bar_steps
        if t_val < 0.5:
            ss = t_val / 0.5
            c = (1.0, 0.3 + ss * 0.6, 0.3)  # red→yellow
        else:
            ss = (t_val - 0.5) / 0.5
            c = (1.0 - ss * 0.85, 0.9 - ss * 0.1, 0.3)  # yellow→green
        rect = plt.Rectangle((bx0, bar_y_bottom), bx1 - bx0, bar_height,
                             transform=ax.transAxes, color=c, linewidth=0, clip_on=False)
        ax.add_patch(rect)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect(aspect / (FIG_W / FIG_H))
    ax.axis('off')

    plt.tight_layout(pad=0.5)

    # Render to PIL Image
    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(img_arr)
    frames.append(img)
    plt.close(fig)

    if (frame_idx + 1) % 10 == 0:
        print(f"  {frame_idx + 1}/{n_frames}")

# ---- Save GIF ----
print(f"\nSaving to {OUTPUT}...")
frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),
    loop=0,
    optimize=True,
)
fsize_mb = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"Done! {OUTPUT}  ({fsize_mb:.1f} MB, {n_frames} frames, {FPS} fps, {n_frames/FPS:.1f}s)")
