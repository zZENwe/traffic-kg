"""Compute per-sensor traffic trends from METR-LA raw data and write to Neo4j."""
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

URI = NEO4J_URI
AUTH = NEO4J_AUTH

print("Loading METR-LA raw data...")
f = h5py.File('data/metr-la.h5', 'r')
vals = f['data/block0_values'][:]  # (34272, 207) float32
times_raw = f['data/axis1'][:]      # timestamps
f.close()

num_steps, num_sensors = vals.shape
print(f"Data: {num_steps} steps x {num_sensors} sensors")

# Parse timestamps (nanoseconds since epoch)
times_ns = times_raw.astype(np.int64)
if times_ns[0] > 1e15:  # looks like nanoseconds
    times_s = times_ns / 1e9
else:
    times_s = times_ns

dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times_s]
hours = np.array([d.hour for d in dts])
weekdays = np.array([d.weekday() for d in dts])  # 0=Mon..6=Sun
tf = f"{dts[0].strftime('%Y-%m-%d')} ~ {dts[-1].strftime('%Y-%m-%d')}"
print(f"Time range: {tf}, interval={(times_s[1]-times_s[0])/60:.0f}min")

# Masks: (num_steps,) boolean arrays
valid = vals > 0
is_congested = vals < 45
is_weekday = weekdays < 5
is_weekend = weekdays >= 5
is_morning = (hours >= 7) & (hours <= 9)
is_evening = (hours >= 16) & (hours <= 18)
is_offpeak = (hours >= 10) & (hours <= 15) | (hours >= 20) | (hours <= 5)

print("\nComputing per-sensor trends (vectorized)...")

# --- Vectorized trend computation (N steps x M sensors) ---
def masked_mean(v, mask_2d):
    """Compute column-wise mean where mask is True; returns (M,) array."""
    masked = np.where(mask_2d, v, np.nan)
    return np.nanmean(masked, axis=0)

avg_all = masked_mean(vals, valid)
avg_morning = masked_mean(vals, valid & is_morning[:, None])
avg_evening = masked_mean(vals, valid & is_evening[:, None])
avg_offpeak = masked_mean(vals, valid & is_offpeak[:, None])
avg_weekday = masked_mean(vals, valid & is_weekday[:, None])
avg_weekend = masked_mean(vals, valid & is_weekend[:, None])
cong_ratio = np.nanmean(np.where(vals > 0, vals < 45, np.nan), axis=0)
peak_drop = np.round(avg_offpeak - np.minimum(avg_morning, avg_evening), 1)

# Hourly profile: find worst hour per sensor
worst_hour = np.zeros(num_sensors, dtype=int)
worst_speed = np.zeros(num_sensors)
for i in range(num_sensors):
    hr_avgs = {}
    for h in range(24):
        mask = valid[:, i] & (hours == h)
        if mask.sum() > 10:
            hr_avgs[h] = float(vals[mask, i].mean())
    if hr_avgs:
        wh = min(hr_avgs, key=hr_avgs.get)
        worst_hour[i] = wh
        worst_speed[i] = hr_avgs[wh]

print(f"\nAvg speed: {np.nanmean(avg_all):.1f} mph")
print(f"Avg congestion ratio: {np.nanmean(cong_ratio):.2%}")
print(f"Avg peak drop: {np.nanmean(peak_drop):.1f} mph")

# Build batch for UNWIND write
sensors_df = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
sensor_ids = [int(sensors_df.iloc[i]['sensor_id']) for i in range(num_sensors)]

batch = []
for i in range(num_sensors):
    batch.append({
        'sid': sensor_ids[i],
        'avg': round(float(avg_all[i]), 1),
        'cr': round(float(cong_ratio[i]), 3),
        'ma': round(float(avg_morning[i]), 1),
        'ea': round(float(avg_evening[i]), 1),
        'oa': round(float(avg_offpeak[i]), 1),
        'pd': round(float(peak_drop[i]), 1),
        'wa': round(float(avg_weekday[i]), 1),
        'we': round(float(avg_weekend[i]), 1),
        'wh': int(worst_hour[i]),
        'ws': round(float(worst_speed[i]), 1),
    })

# Write to Neo4j in a single batch
print("\nWriting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

with driver.session() as session:
    session.run("""
        UNWIND $batch AS data
        MATCH (s:Sensor {sid: data.sid})
        SET s.avg_speed = data.avg,
            s.congestion_ratio = data.cr,
            s.morning_avg = data.ma,
            s.evening_avg = data.ea,
            s.offpeak_avg = data.oa,
            s.peak_drop = data.pd,
            s.weekday_avg = data.wa,
            s.weekend_avg = data.we,
            s.worst_hour = data.wh,
            s.worst_speed = data.ws
    """, batch=batch)
    print(f"  Updated {num_sensors} sensors with trend data")

    # Verify sample
    r = session.run("""
        MATCH (s:Sensor)
        RETURN avg(s.avg_speed) as a, avg(s.congestion_ratio) as c,
               avg(s.morning_avg) as m, avg(s.evening_avg) as e,
               avg(s.offpeak_avg) as o, avg(s.weekday_avg) as wd,
               avg(s.weekend_avg) as we, avg(s.peak_drop) as pd
    """).single()
    print(f"  Verified: avg_speed={r['a']:.1f}, cong={r['c']:.3f}, "
          f"morning={r['m']:.1f}, evening={r['e']:.1f}, offpeak={r['o']:.1f}, "
          f"weekday={r['wd']:.1f}, weekend={r['we']:.1f}, peak_drop={r['pd']:.1f}")

driver.close()
print("Done!")
