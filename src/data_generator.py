"""
Data Generator Module for Public Transit Delays and Weather Context
Simulates realistic urban transit operations (GTFS-style logs) and synchronized weather data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional


def generate_synthetic_transit_dataset(
    start_date: str = "2026-01-01",
    days: int = 90,
    routes: Optional[List[str]] = None,
    stops_per_route: int = 8,
    frequency_minutes: int = 15,
    gps_dropout_rate: float = 0.05,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a realistic synthetic GTFS transit log dataset along with synchronized weather data.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    days : int
        Number of days to simulate.
    routes : List[str]
        List of route identifiers. Defaults to ['Route_101_Uptown', 'Route_102_Crosstown', 'Route_204_Downtown', 'Route_500_Express'].
    stops_per_route : int
        Number of sequential stops along each route.
    frequency_minutes : int
        Scheduled headway between vehicle trips in minutes.
    gps_dropout_rate : float
        Probability of simulated GPS sensor signal loss / missing record.
    random_seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - transit_df: Raw transit logs with timestamps, RouteID, StopID, VehicleID, Delay_Seconds, Delay_Minutes.
        - weather_df: Synchronized weather observations.
    """
    np.random.seed(random_seed)
    
    if routes is None:
        routes = [
            "Route_101_Uptown",
            "Route_102_Crosstown",
            "Route_204_Downtown",
            "Route_500_Express"
        ]
        
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + timedelta(days=days)
    
    # 1. Generate Synchronized Weather Data at hourly intervals
    hourly_timestamps = pd.date_range(start=start_dt, end=end_dt, freq="1h")
    n_hours = len(hourly_timestamps)
    
    # Simulate seasonal and diurnal temperature cycle (°C)
    base_temp = 8.0 + 6.0 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365))  # seasonal
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (hourly_timestamps.hour - 6) / 24)  # diurnal
    temp_noise = np.random.normal(0, 1.5, n_hours)
    temperature = np.round(base_temp + diurnal_temp + temp_noise, 1)
    
    # Simulate weather conditions and precipitation
    # Markov chain-like rain events
    rain_prob = 0.12
    is_rain = np.zeros(n_hours, dtype=bool)
    in_storm = False
    storm_duration = 0
    
    precipitation = np.zeros(n_hours, dtype=float)
    wind_speed = np.random.gamma(shape=3.0, scale=2.5, size=n_hours)
    
    for i in range(n_hours):
        if in_storm:
            storm_duration -= 1
            is_rain[i] = True
            precipitation[i] = np.random.uniform(1.5, 12.0)
            wind_speed[i] += np.random.uniform(5.0, 15.0)
            if storm_duration <= 0:
                in_storm = False
        else:
            if np.random.rand() < rain_prob:
                in_storm = True
                storm_duration = np.random.randint(2, 9)  # storm lasts 2-8 hours
                is_rain[i] = True
                precipitation[i] = np.random.uniform(0.5, 6.0)
                
    # Classify weather condition
    conditions = []
    for i in range(n_hours):
        p = precipitation[i]
        t = temperature[i]
        if p > 6.0:
            conditions.append("Heavy Rain" if t > 0 else "Heavy Snow")
        elif p > 0.5:
            conditions.append("Light Rain" if t > 0 else "Snow")
        elif wind_speed[i] > 25.0:
            conditions.append("High Wind")
        elif np.random.rand() < 0.25:
            conditions.append("Cloudy")
        else:
            conditions.append("Clear")
            
    weather_df = pd.DataFrame({
        "timestamp": hourly_timestamps,
        "temperature_c": temperature,
        "precipitation_mm": np.round(precipitation, 2),
        "wind_speed_kmh": np.round(wind_speed, 1),
        "weather_condition": conditions
    })
    
    # Create fast lookup map for weather by floor hour
    weather_lookup = weather_df.set_index("timestamp")
    
    # 2. Generate Transit Trips and Sequential Stop Logs
    records = []
    
    # Vehicle pool per route
    vehicle_pools = {
        route: [f"BUS_{route.split('_')[1]}_{v:02d}" for v in range(1, 15)]
        for route in routes
    }
    
    # Simulate day by day
    for day_idx in range(days):
        current_day = start_dt + timedelta(days=day_idx)
        is_weekend = current_day.weekday() >= 5
        
        # Determine holiday (e.g. day 15 or 45 or Jan 1st)
        is_holiday = (current_day.month == 1 and current_day.day == 1) or (day_idx in [15, 45, 75])
        
        # Operational hours: 05:00 to 23:30
        trip_start_times = pd.date_range(
            start=current_day.replace(hour=5, minute=0, second=0),
            end=current_day.replace(hour=23, minute=30, second=0),
            freq=f"{frequency_minutes}min"
        )
        
        for trip_idx, trip_start in enumerate(trip_start_times):
            hour = trip_start.hour + trip_start.minute / 60.0
            
            # Rush hour multiplier
            # Morning peak: 07:30 to 09:30 (peak at 8:15)
            # Evening peak: 16:30 to 19:00 (peak at 17:30)
            morning_rush = np.exp(-((hour - 8.25) ** 2) / (2 * (0.9 ** 2)))
            evening_rush = np.exp(-((hour - 17.5) ** 2) / (2 * (1.1 ** 2)))
            rush_intensity = 1.0 + (2.5 * morning_rush + 3.0 * evening_rush) * (0.35 if is_weekend else 1.0)
            
            # Weather lookup
            weather_hour = trip_start.floor("1h")
            if weather_hour in weather_lookup.index:
                w_row = weather_lookup.loc[weather_hour]
                precip = w_row["precipitation_mm"]
                wind = w_row["wind_speed_kmh"]
                temp = w_row["temperature_c"]
            else:
                precip, wind, temp = 0.0, 10.0, 15.0
                
            # Weather delay factor: rain & freezing temps increase friction & dwell time
            weather_delay_factor = 1.0 + (precip * 0.45) + (max(0, 20.0 - wind) * 0.0) + (1.5 if temp < 0 else 0.0)
            
            for route_idx, route in enumerate(routes):
                vehicle_id = vehicle_pools[route][trip_idx % len(vehicle_pools[route])]
                
                # Base cumulative delay accumulator for the trip
                # Some trips start slightly delayed or on time
                accumulated_delay_sec = np.random.normal(30, 25)
                
                # Route specific factor (Express has fewer stops but higher highway variance)
                route_factor = 1.3 if "Express" in route else (1.1 if "Downtown" in route else 0.9)
                
                # Inter-stop scheduled travel time: ~4 minutes per stop
                for stop_idx in range(1, stops_per_route + 1):
                    stop_id = f"STOP_{route.split('_')[1]}_{stop_idx:02d}"
                    scheduled_travel_sec = (stop_idx - 1) * 240
                    scheduled_arrival = trip_start + timedelta(seconds=scheduled_travel_sec)
                    
                    # Segment delay addition:
                    # Depends on rush hour, weather, route factor, and random traffic fluctuation
                    traffic_noise = np.random.exponential(scale=35.0)
                    dwell_delay = np.random.gamma(shape=2.0, scale=15.0) * rush_intensity
                    
                    segment_delay = (
                        (15.0 * rush_intensity * route_factor * weather_delay_factor)
                        + traffic_noise
                        + dwell_delay
                    )
                    
                    # Occasional random road incident / signal failure (1.5% probability)
                    if np.random.rand() < 0.015:
                        segment_delay += np.random.uniform(180, 600)  # +3 to +10 mins
                        
                    accumulated_delay_sec += segment_delay
                    
                    # Clamp lower bound at -60s (buses rarely run more than 1 min early)
                    accumulated_delay_sec = max(-60.0, accumulated_delay_sec)
                    
                    actual_arrival = scheduled_arrival + timedelta(seconds=accumulated_delay_sec)
                    delay_seconds = int(accumulated_delay_sec)
                    delay_minutes = round(delay_seconds / 60.0, 2)
                    
                    # Simulate GPS sensor dropouts
                    if np.random.rand() < gps_dropout_rate:
                        continue  # Lost GPS packet
                        
                    records.append({
                        "timestamp": actual_arrival,
                        "scheduled_arrival": scheduled_arrival,
                        "route_id": route,
                        "stop_id": stop_id,
                        "vehicle_id": vehicle_id,
                        "delay_seconds": delay_seconds,
                        "delay_minutes": delay_minutes,
                        "is_weekend": int(is_weekend),
                        "is_holiday": int(is_holiday)
                    })
                    
    transit_df = pd.DataFrame(records)
    transit_df.sort_values(by="timestamp", inplace=True)
    transit_df.reset_index(drop=True, inplace=True)
    
    return transit_df, weather_df


def save_datasets(
    transit_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: str = "data/raw"
) -> Tuple[str, str]:
    """
    Saves generated datasets to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    transit_path = os.path.join(output_dir, "gtfs_delay_logs.csv")
    weather_path = os.path.join(output_dir, "weather_observations.csv")
    
    transit_df.to_csv(transit_path, index=False)
    weather_df.to_csv(weather_path, index=False)
    
    return transit_path, weather_path


if __name__ == "__main__":
    print("Generating synthetic GTFS transit logs and weather dataset...")
    transit_df, weather_df = generate_synthetic_transit_dataset(
        start_date="2026-01-01",
        days=90,
        frequency_minutes=15,
        gps_dropout_rate=0.04
    )
    print(f"Generated {len(transit_df):,} transit logs and {len(weather_df):,} weather records.")
    t_path, w_path = save_datasets(transit_df, weather_df)
    print(f"Saved transit logs to: {t_path}")
    print(f"Saved weather observations to: {w_path}")
