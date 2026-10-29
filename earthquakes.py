#####################################
#%% Setup
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Download latest version
#path = kagglehub.dataset_download("alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023")
#print("Path to dataset files:", path)

# Hardcoding data source
data = r"C:\Users\nansh\.cache\kagglehub\datasets\alessandrolobello\the-ultimate-earthquake-dataset-from-1990-2023\versions\4\Eartquakes-1990-2023.csv"
df = pd.read_csv(f"{data}")
df["time"] = pd.to_datetime(df["time"], unit="ms")
df.rename(columns={"magnitudo": "magnitude"}, inplace=True)
#####################################
## Histogram
#%% Basic analyis 
print(df.describe())
df_large_mag = df[df["magnitude"] > 0]["magnitude"]
df_large_mag.plot.hist(bins=100, grid=True, log=False, title="Earthquakes from 1990-2023")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")

plt.show()
#####################################
## Map arrangement by magnitude
#%% Global mapping
# Define the magnitude threshold
threshold = 5.5

# Set up df
df_loc = df[df["magnitude"] > threshold][["magnitude", "longitude", "latitude", "time"]]
df_loc["year"] = df_loc["time"].dt.year
# Graph it on a map
fig_map = px.scatter_geo(df_loc, lat="latitude", lon="longitude", 
                         hover_name="magnitude", hover_data=["year"],
                        title=f"EQs mag > {threshold} since 1990", color="magnitude")

# Reduce amount of whitespace
fig_map.update_layout(margin=dict(l=20, r=20))
fig_map.show()

#####################################
## Basic ML code/ideas
#%% Machine Learning
main_df = df[df["magnitude"]>7][["time", "magnitude"]]
main_df.head()

## Plan
# Run a k-nearest neighbour to figure out tectonic plates
# Then for each group, run an analysis to find the average waiting time for each group
# Find and backtest the three results generally
# Try and predict if there will be an eq > 7 before the end may and
# assign an appropriate probability
