"""Fetch historical LA weather and compute per-sensor weather impact."""
import requests
import pandas as pd
import numpy as np
import h5py
from datetime import datetime, timezone

print("[1/4] Fetching historical weather from Open-Meteo...")
# LA area: 34.05, -118.25. 2012-03-01 to 2012-06-27
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.05,
    "longitude": -118.25,
    "start_date": "2012-03-01",
    "end_date": "2012-06-27",
    "hourly": ["precipitation", "temperature_2m", "wind_speed_10m", "cloud_cover"],
    "timezone": "America/Los_Angeles",
}
r = requests.get(url, params=params, timeout=30)
data = r.json()
hourly = data["hourly"]
weather = pd.DataFrame({
    "time": pd.to_datetime(hourly["time"]),
    "precip_mm": hourly["precipitation"],
    "temp_c": hourly["temperature_2m"],
    "wind_kmh": hourly["wind_speed_10m"],
    "cloud_pct": hourly["cloud_cover"],
})
weather["is_rain"] = weather["precip_mm"] > 0.1
weather["is_heavy_rain"] = weather["precip_mm"] > 2.0

rain_hours = weather["is_rain"].sum()
total_hours = len(weather)
print(f"  {total_hours} hours fetched, {rain_hours} rainy ({rain_hours/total_hours*100:.1f}%)")
print(f"  Temp range: {weather['temp_c'].min():.0f}-{weather['temp_c'].max():.0f}C")
print(f"  Heavy rain hours: {weather['is_heavy_rain'].sum()}")

# Save weather data
weather.to_csv("data/weather_2012.csv", index=False)

# [2/4] Match weather to METR-LA timestamps
print("\n[2/4] Matching weather to METR-LA 5-min intervals...")
f = h5py.File("data/metr-la.h5", "r")
vals = f["data/block0_values"][:]
times_ns = f["data/axis1"][:].astype(np.int64)
f.close()

times_s = times_ns / 1e9
dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times_s]
num_steps, num_sensors = vals.shape
print(f"  {num_steps} time steps, {num_sensors} sensors")

# For each 5-min step, find the nearest hourly weather record
weather_times_ns = weather["time"].values.astype("datetime64[h]").astype(np.int64)
weather_indices = []
for dt in dts:
    dt_hour = np.datetime64(dt.replace(minute=0, second=0, microsecond=0, tzinfo=None), "h")
    hour_idx = np.argmin(np.abs(weather_times_ns - dt_hour.astype(np.int64)))
    weather_indices.append(hour_idx)

weather_indices = np.array(weather_indices)

# [3/4] Compute per-sensor weather impact
print("\n[3/4] Computing per-sensor weather impact...")
impacts = {}
for i in range(num_sensors):
    v = vals[:, i]
    valid = v > 0

    rain_mask = valid & weather["is_rain"].values[weather_indices]
    clear_mask = valid & ~weather["is_rain"].values[weather_indices]
    heavy_mask = valid & weather["is_heavy_rain"].values[weather_indices]

    rain_avg = float(v[rain_mask].mean()) if rain_mask.sum() > 10 else 0
    clear_avg = float(v[clear_mask].mean()) if clear_mask.sum() > 10 else 0
    heavy_avg = float(v[heavy_mask].mean()) if heavy_mask.sum() > 5 else 0

    rain_delta = round(clear_avg - rain_avg, 1) if rain_avg > 0 and clear_avg > 0 else 0
    heavy_delta = round(clear_avg - heavy_avg, 1) if heavy_avg > 0 and clear_avg > 0 else 0

    # Rain congestion increase
    rain_cong = 0.0
    if rain_mask.sum() > 10 and clear_mask.sum() > 10:
        rain_cong_pct = float((v[rain_mask] < 45).mean()) if rain_mask.sum() > 10 else 0
        clear_cong_pct = float((v[clear_mask] < 45).mean()) if clear_mask.sum() > 10 else 0
        rain_cong = round((rain_cong_pct - clear_cong_pct) * 100, 1)

    impacts[i] = {
        "rain_speed": round(rain_avg, 1),
        "clear_speed": round(clear_avg, 1),
        "rain_speed_drop": rain_delta,
        "heavy_rain_drop": heavy_delta,
        "rain_congestion_increase": rain_cong,
        "rain_hours": int(rain_mask.sum()),
    }

    if i % 50 == 0:
        print(f"  {i}/{num_sensors}...")

drops = [imp["rain_speed_drop"] for imp in impacts.values()]
incs = [imp["rain_congestion_increase"] for imp in impacts.values()]
print(f"\n  Avg speed drop in rain: {np.mean([d for d in drops if d > 0]):.1f} mph")
print(f"  Avg congestion increase: {np.mean([c for c in incs if c > 0]):.1f}%")
print(f"  Sensors with rain data: {sum(1 for imp in impacts.values() if imp['rain_hours'] > 10)}")

# [4/4] Write to Neo4j
print("\n[4/4] Writing weather impact to Neo4j...")
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

sensors_csv = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
sensor_ids = [int(sensors_csv.iloc[i]["sensor_id"]) for i in range(num_sensors)]

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
driver.verify_connectivity()

with driver.session() as session:
    for idx, sid in enumerate(sensor_ids):
        imp = impacts[idx]
        session.run("""
            MATCH (s:Sensor {sid: $sid})
            SET s.rain_speed_drop = $rsd,
                s.rain_congestion_increase = $rci,
                s.heavy_rain_drop = $hrd,
                s.rain_hours = $rh
        """, sid=sid, rsd=imp["rain_speed_drop"], rci=imp["rain_congestion_increase"],
             hrd=imp["heavy_rain_drop"], rh=imp["rain_hours"])

    # Verify
    r = session.run("""
        MATCH (s:Sensor) WHERE s.rain_speed_drop IS NOT NULL
        RETURN count(s) as cnt, avg(s.rain_speed_drop) as avg_drop,
               avg(s.rain_congestion_increase) as avg_ci
    """).single()
    print(f"  Written: {r['cnt']} sensors, avg rain drop={r['avg_drop']:.1f}mph, "
          f"congestion +{r['avg_ci']:.1f}%")

    # Top rain-affected
    print("\n  Top 5 rain-affected sensors:")
    top = session.run("""
        MATCH (s:Sensor) WHERE s.rain_hours > 10
        RETURN s.sid as sid, s.rain_speed_drop as drop,
               s.rain_congestion_increase as ci, s.road_type as rt
        ORDER BY s.rain_speed_drop DESC LIMIT 5
    """).data()
    for t in top:
        print(f"    Sensor {t['sid']}: -{t['drop']}mph, congestion +{t['ci']}%, {t['rt']}")

driver.close()
print("\nDone!")
