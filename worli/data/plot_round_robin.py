import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
traffic_df = pd.read_csv('../round_robin/rr_traffic.csv')
trip_df = pd.read_csv('../round_robin/rr_trip.csv')

# Define constants
episode_sec = 3600  # Simulation duration
window_size = 300  # 5-minute window for trip completion rate

# Function for moving average smoothing
def moving_average(series, window):
    return series.rolling(window, min_periods=1).mean()

# Plot function
def plot_series(df, x_col, y_col, xlabel, ylabel, title, filename, agg='mv', window=None):
    plt.figure(figsize=(9, 6))
    
    x = df[x_col].values
    y = df[y_col].values
    
    if agg == 'mv' and window:
        y = moving_average(pd.Series(y), window).values

    plt.plot(x, y, linewidth=2, label=y_col)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("../round_robin/{}.png".format(filename))
    plt.close()

# Generate plots
plot_series(traffic_df, 'time_sec', 'avg_speed_mps', 'Simulation Time (s)', 'Average Car Speed (m/s)',
            'Average Car Speed Over Time', 'avg_speed', agg='mv', window=60)

plot_series(traffic_df, 'time_sec', 'number_arrived_car', 'Simulation Time (s)', 'Trip Completion Rate (veh/5min)',
            'Trip Completion Rate Over Time', 'trip_completion', agg='sum', window=window_size)

plot_series(trip_df, 'arrival_sec', 'wait_sec', 'Simulation Time (s)', 'Average Trip Delay (s)',
            'Average Trip Delay Over Time', 'trip_delay', agg='mean', window=60)

plot_series(traffic_df, 'time_sec', 'avg_wait_sec', 'Simulation Time (s)', 'Average Intersection Delay (s/veh)',
            'Average Intersection Delay Over Time', 'intersection_delay', agg='mv', window=60)
