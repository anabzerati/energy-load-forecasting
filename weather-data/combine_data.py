import pandas as pd
import numpy as np

CONSUMPTION_PATH = "../Data Morocco - Laayoune.csv"
WEATHER_PATH = "hourly_weather_data.csv"   # or "daily_weather_data.csv"
OUTPUT_PATH = "consumption_weather_hourly.csv"  # or "consumption_weather_daily.csv"
TIMEZONE = "Africa/Casablanca"  # Local timezone for Laayoune

# Laayoune consumption data
consumption = pd.read_csv(
    CONSUMPTION_PATH,
    sep=",",
    decimal=",",
    parse_dates=["DateTime"],
)

consumption.rename(columns={"DateTime": "date", "zone1": "consumption"}, inplace=True)
consumption = consumption.sort_values("date")

# hourly aggregation 
consumption_hourly = (
    consumption.set_index("date")
    .resample("1h")["consumption"]
    .sum()
    .reset_index()
)

# daily aggregation 
# consumption_daily = (
#     consumption.set_index("date")
#     .resample("1d")["consumption"]
#     .sum()
#     .reset_index()
# )

# timezone 
consumption_hourly["date"] = consumption_hourly["date"].dt.tz_localize(
    TIMEZONE, ambiguous="NaT", nonexistent="NaT"
)

# weather data
weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
weather["date"] = weather["date"].dt.tz_convert(TIMEZONE)


# merge
merged = pd.merge(
    consumption_hourly.sort_values("date"),
    weather.sort_values("date"),
    on="date",
    how="inner"  # matching timestamps only
)

merged.to_csv(OUTPUT_PATH, index=False)
print(merged.head())