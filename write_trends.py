"""Compute per-sensor traffic trends from METR-LA raw data and write to Neo4j."""
import h5py
import numpy as np
from datetime import datetime, timezone
from neo4j import GraphDatabase

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

# Masks
is_weekday = weekdays < 5
is_weekend = weekdays >= 5
is_morning = (hours >= 7) & (hours <= 9)
is_evening = (hours >= 16) & (hours <= 18)
is_offpeak = (hours >= 10) & (hours <= 15) | (hours >= 20) | (hours <= 5)
is_congested = vals < 45  # mph below 45 = congested

print("\nComputing per-sensor trends...")
trends = {}
for i in range(num_sensors):
    v = vals[:, i]
    valid = v > 0

    def safe_mean(mask):
        s = v[mask]
        return float(s.mean()) if len(s) > 0 else 0.0

    avg_all = safe_mean(valid)
    avg_morning = safe_mean(valid & is_morning)
    avg_evening = safe_mean(valid & is_evening)
    avg_offpeak = safe_mean(valid & is_offpeak)
    avg_weekday = safe_mean(valid & is_weekday)
    avg_weekend = safe_mean(valid & is_weekend)
    cong_ratio = float(((v > 0) & (v < 45)).mean().item())
    peak_drop = round(avg_offpeak - min(avg_morning, avg_evening), 1)

    # Hourly profile: find worst hour
    hr_avgs = {}
    for h in range(24):
        mask = valid & (hours == h)
        if mask.sum() > 10:
            hr_avgs[h] = float(v[mask].mean())
    worst_hour = min(hr_avgs, key=hr_avgs.get) if hr_avgs else -1
    worst_speed = hr_avgs.get(worst_hour, 0)

    trends[i] = {
        'avg_speed': round(avg_all, 1),
        'congestion_ratio': round(cong_ratio, 3),
        'morning_avg': round(avg_morning, 1),
        'evening_avg': round(avg_evening, 1),
        'offpeak_avg': round(avg_offpeak, 1),
        'peak_drop': peak_drop,
        'weekday_avg': round(avg_weekday, 1),
        'weekend_avg': round(avg_weekend, 1),
        'worst_hour': worst_hour,
        'worst_speed': round(worst_speed, 1),
    }

    if i % 50 == 0:
        print(f"  {i}/{num_sensors}...")

print(f"\nAvg speed: {np.mean([t['avg_speed'] for t in trends.values()]):.1f} mph")
print(f"Avg congestion ratio: {np.mean([t['congestion_ratio'] for t in trends.values()]):.2%}")
print(f"Avg peak drop: {np.mean([t['peak_drop'] for t in trends.values()]):.1f} mph")

# Write to Neo4j
print("\nWriting to Neo4j...")
import pandas as pd
from config import NEO4J_URI, NEO4J_AUTH
sensors = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
sensor_ids = [int(sensors.iloc[i]['sensor_id']) for i in range(num_sensors)]

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

with driver.session() as session:
    for idx, sid in enumerate(sensor_ids):
        t = trends[idx]
        session.run("""
            MATCH (s:Sensor {sid: $sid})
            SET s.avg_speed = $avg,
                s.congestion_ratio = $cr,
                s.morning_avg = $ma,
                s.evening_avg = $ea,
                s.offpeak_avg = $oa,
                s.peak_drop = $pd,
                s.weekday_avg = $wa,
                s.weekend_avg = $we,
                s.worst_hour = $wh,
                s.worst_speed = $ws
        """, sid=sid, avg=t['avg_speed'], cr=t['congestion_ratio'],
             ma=t['morning_avg'], ea=t['evening_avg'], oa=t['offpeak_avg'],
             pd=t['peak_drop'], wa=t['weekday_avg'], we=t['weekend_avg'],
             wh=t['worst_hour'], ws=t['worst_speed'])

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
