"""Generate animated flow GIF — heatmap + flowing directional edges.

Each edge drawn as gradient line: thick/bright at fast end, thin/dim at slow end.
Creates a fluid "flowing network" effect.
"""
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image
import os
from datetime import datetime, timezone

OUTPUT = 'outputs/traffic_flow.gif'
FPS = 8
SAMPLE_STEP = 12
MAX_FRAMES = 72
DPI = 150
FIG_W, FIG_H = 16, 7
FLOW_THRESHOLD = 3.0   # mph — min diff to show flow
FLOW_MAX = 20.0         # mph — diff at which flow is fully opaque

base = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(base, 'data', 'metr-la.h5')
csv_path = os.path.join(base, 'data', 'sensor_graph', 'graph_sensor_locations.csv')
os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)

print("Loading METR-LA data...")
with h5py.File(h5_path, 'r') as f:
    vals = f['data/block0_values'][:]
    times_raw = f['data/axis1'][:]

times_ns = times_raw.astype(np.int64)
times_s = times_ns / 1e9 if times_ns[0] > 1e15 else times_ns
vals_ds = vals[::SAMPLE_STEP, :][:MAX_FRAMES]
times_ds = times_s[::SAMPLE_STEP][:MAX_FRAMES]
timestamps = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%m/%d %H:%M') for t in times_ds]
n_frames, n_sensors = vals_ds.shape

sensors = pd.read_csv(csv_path, index_col=0)
lats = sensors['latitude'].values[:n_sensors]
lons = sensors['longitude'].values[:n_sensors]
lon_range = lons.max() - lons.min()
lat_range = lats.max() - lats.min()
aspect = lon_range / lat_range

x = (lons - lons.min()) / lon_range
y = (lats.max() - lats) / lat_range
y = 1 - y
x = (x - 0.5) * 1.4 + 0.5

# Build edge list (undirected pairs)
coords = sensors[['latitude', 'longitude']].values
geo_dist = np.sqrt((coords[:, None, 0] - coords[None, :, 0]) ** 2 +
                   (coords[:, None, 1] - coords[None, :, 1]) ** 2)
edge_pairs = []
for i in range(n_sensors):
    for j in np.argsort(geo_dist[i])[1:6]:  # top-5 nearest
        if j < i:
            edge_pairs.append((j, i))
        elif (j, i) not in edge_pairs:
            edge_pairs.append((i, j))
print(f"  {n_frames} frames, {n_sensors} sensors, {len(edge_pairs)} edges")


def speed_color_rgba(spd, alpha=0.85):
    if spd <= 0: return (0.3, 0.3, 0.3, 0.3)
    t = np.clip(spd / 70, 0, 1)
    if t < 0.5:
        s = t / 0.5
        return (1.0, 0.3 + s * 0.6, 0.3, alpha)
    else:
        s = (t - 0.5) / 0.5
        return (1.0 - s * 0.85, 0.9 - s * 0.1, 0.3, alpha)


def draw_flow_edge(ax, x0, y0, x1, y1, diff, spd_fast, spd_slow):
    """Draw tapered wedge with gradient color from fast→slow, matching heatmap."""
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-8:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular

    intensity = min(1.0, (diff - FLOW_THRESHOLD) / (FLOW_MAX - FLOW_THRESHOLD))
    segments = 6
    w_start = 0.001 + intensity * 0.005
    w_end = 0.0005 + intensity * 0.0015

    for k in range(segments):
        t0 = k / segments
        t1 = (k + 1) / segments

        # Position along edge
        sx0, sy0 = x0 + ux * t0 * length, y0 + uy * t0 * length
        sx1, sy1 = x0 + ux * t1 * length, y0 + uy * t1 * length

        # Width at this segment (linear taper)
        w0 = w_start + (w_end - w_start) * t0
        w1 = w_start + (w_end - w_start) * t1

        # Color: blend fast sensor color → slow sensor color along t
        t_mid = (t0 + t1) / 2
        c_heavy = speed_color_rgba(spd_fast, alpha=1.0)
        c_light = speed_color_rgba(spd_slow, alpha=1.0)
        cr = c_heavy[0] + (c_light[0] - c_heavy[0]) * t_mid
        cg = c_heavy[1] + (c_light[1] - c_heavy[1]) * t_mid
        cb = c_heavy[2] + (c_light[2] - c_heavy[2]) * t_mid
        alpha = (0.12 + intensity * 0.55) * (1.0 - t_mid * 0.5)
        seg_color = (cr, cg, cb, min(1.0, alpha))

        vertices = [
            (sx0 + px * w0, sy0 + py * w0),
            (sx0 - px * w0, sy0 - py * w0),
            (sx1 - px * w1, sy1 - py * w1),
            (sx1 + px * w1, sy1 + py * w1),
        ]
        poly = Polygon(vertices, facecolor=seg_color, edgecolor='none', zorder=3)
        ax.add_patch(poly)


print(f"Rendering {n_frames} frames...")
frames = []

for frame_idx in range(n_frames):
    speeds = vals_ds[frame_idx, :]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    # 1. Background road edges (faint)
    bg_segs = [[(x[i], y[i]), (x[j], y[j])] for i, j in edge_pairs]
    lc = LineCollection(bg_segs, colors='#0f3460', linewidths=0.3, alpha=0.35)
    ax.add_collection(lc)

    # 2. Flow wedges — draw thicker ones on top for depth
    flows = []
    for i, j in edge_pairs:
        si, sj = speeds[i], speeds[j]
        if si <= 0 or sj <= 0:
            continue
        diff = si - sj
        if abs(diff) < FLOW_THRESHOLD:
            continue
        if diff > 0:
            flows.append((x[i], y[i], x[j], y[j], diff, si, sj))
        else:
            flows.append((x[j], y[j], x[i], y[i], -diff, sj, si))

    # Sort by intensity so brighter flows render on top
    flows.sort(key=lambda f: f[4])

    for fx0, fy0, fx1, fy1, fdiff, ffast, fslow in flows:
        draw_flow_edge(ax, fx0, fy0, fx1, fy1, fdiff, ffast, fslow)

    # 3. Sensor nodes on top
    sizes = np.where(speeds > 0, 18 + (1 - np.clip(speeds / 70, 0, 1)) * 45, 6)
    colors = np.zeros((n_sensors, 4))
    for i in range(n_sensors):
        colors[i] = speed_color_rgba(speeds[i])
    ax.scatter(x, y, s=sizes, c=colors, edgecolors='white', linewidths=0.12, zorder=10)

    # Title
    ax.text(0.5, 1.02, 'LA Highway Traffic — Dynamic Flow Network',
            transform=ax.transAxes, ha='center', fontsize=14, color='#e94560', fontweight='bold')
    ax.text(0.5, 0.97, f'METR-LA · 207 Sensors · {timestamps[frame_idx]} · Gradient wedges = flow direction',
            transform=ax.transAxes, ha='center', fontsize=10, color='#888')

    # Legend
    for label, c, sx in [('Congested', '#ef4444', 0.20), ('Medium', '#eab308', 0.42), ('Free Flow', '#22c55e', 0.65)]:
        ax.text(sx, 0.05, label, transform=ax.transAxes, fontsize=8, color=c, ha='center')

    # Gradient bar
    for k in range(80):
        bx0 = 0.15 + k / 80 * 0.70
        bx1 = 0.15 + (k + 1) / 80 * 0.70
        t_val = k / 80
        if t_val < 0.5:
            ss = t_val / 0.5
            c = (1.0, 0.3 + ss * 0.6, 0.3)
        else:
            ss = (t_val - 0.5) / 0.5
            c = (1.0 - ss * 0.85, 0.9 - ss * 0.1, 0.3)
        rect = plt.Rectangle((bx0, 0.01), bx1 - bx0, 0.025,
                             transform=ax.transAxes, color=c, linewidth=0, clip_on=False)
        ax.add_patch(rect)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect(aspect / (FIG_W / FIG_H))
    ax.axis('off')
    plt.tight_layout(pad=0.5)

    fig.canvas.draw()
    img = Image.fromarray(np.array(fig.canvas.renderer.buffer_rgba()))
    frames.append(img)
    plt.close(fig)

    if (frame_idx + 1) % 10 == 0:
        print(f"  {frame_idx + 1}/{n_frames}")

print(f"\nSaving to {OUTPUT}...")
frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, optimize=True)
fsize_mb = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"Done! {OUTPUT}  ({fsize_mb:.1f} MB, {n_frames} frames, {FPS} fps, {n_frames/FPS:.1f}s)")
