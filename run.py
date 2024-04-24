# Constants
START_YEAR = 1880
END_YEAR = 2023
BASELINE_START_YEAR = 1961
BASELINE_END_YEAR = 1990
EARTH_RADIUS = 6371
NEARBY_STATION_RADIUS = 1200.0

# Data
GHCN_TEMP_URL = "https://data.giss.nasa.gov/pub/gistemp/ghcnm.tavg.qcf.dat"
GHCN_META_URL = "https://data.giss.nasa.gov/pub/gistemp/v4.inv"

import step0, step1, step2
import os
import shutil

# Remove results directory if it exists
results_dir = "results/"
if os.path.exists(results_dir):
    shutil.rmtree("results/")

# Location for intermediate/final results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Step 0
step0_output = step0.step0(GHCN_TEMP_URL, GHCN_META_URL, START_YEAR, END_YEAR)
step0_filename = "step0_output.csv"
step0_filepath = os.path.join(results_dir, step0_filename)
step0_output.to_csv(step0_filepath)

# Execute Step 1
# (Clean data (by coordinates / drop rules file)
step1_output = step1.step1(step0_output, START_YEAR, END_YEAR)
step1_filename = "step1_output.csv"
step1_filepath = os.path.join(results_dir, step1_filename)
step1_output.to_csv(step1_filepath)

# Execute Step 2
# (Create the 2x2 grid)
step2_output = step2.step2(NEARBY_STATION_RADIUS, EARTH_RADIUS)
step2_filename = "step2_output.csv"
step2_filepath = os.path.join(results_dir, step2_filename)
step2_output.to_csv(step2_filepath)
