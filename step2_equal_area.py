import math
import pandas as pd


def calculate_area(row) -> float:
    earth_radius_km: float = 6371.0
    delta_longitude: float = np.radians(row["Eastern"] - row["Western"])
    southern_latitude: float = np.radians(row["Southern"])
    northern_latitude: float = np.radians(row["Northern"])
    area: float = (
        (earth_radius_km**2)
        * delta_longitude
        * (np.sin(northern_latitude) - np.sin(southern_latitude))
    )
    return area


def calculate_center_coordinates(row):
    """Calculate the center latitude and longitude for a given box.

    Args:
        row (pd.Series): A Pandas Series representing a row of the DataFrame with ('southern', 'northern', 'western', 'eastern') coordinates.

    Returns:
        Tuple[float, float]: A tuple containing the center latitude and longitude.
    """
    center_latitude = 0.5 * (
        math.sin(row["Southern"] * math.pi / 180)
        + math.sin(row["Northern"] * math.pi / 180)
    )
    center_longitude = 0.5 * (row["Western"] + row["Eastern"])
    center_latitude = math.asin(center_latitude) * 180 / math.pi
    return center_latitude, center_longitude


def generate_80_cell_grid() -> pd.DataFrame:
    """Generate an 80-cell grid DataFrame with columns for southern, northern, western, eastern,
    center_latitude, and center_longitude coordinates.

    Returns:
        pd.DataFrame: The generated DataFrame.
    """
    grid_data = []

    # Number of horizontal boxes in each band
    # (proportional to the thickness of each band)
    band_boxes = [4, 8, 12, 16]

    # Sines of latitudes
    band_altitude = [1, 0.9, 0.7, 0.4, 0]

    # Generate the 40 cells in the northern hemisphere
    for band in range(len(band_boxes)):
        n = band_boxes[band]
        for i in range(n):
            lats = 180 / math.pi * math.asin(band_altitude[band + 1])
            latn = 180 / math.pi * math.asin(band_altitude[band])
            lonw = -180 + 360 * float(i) / n
            lone = -180 + 360 * float(i + 1) / n
            box = (lats, latn, lonw, lone)
            grid_data.append(box)

    # Generate the 40 cells in the southern hemisphere by reversing the northern hemisphere cells
    for box in grid_data[::-1]:
        grid_data.append((-box[1], -box[0], box[2], box[3]))

    # Create a DataFrame from the grid data
    df = pd.DataFrame(grid_data, columns=["Southern", "Northern", "Western", "Eastern"])

    # Calculate center coordinates for each box and add them as new columns
    center_coords = df.apply(calculate_center_coordinates, axis=1)
    df[["Center_Latitude", "Center_Longitude"]] = pd.DataFrame(
        center_coords.tolist(), index=df.index
    )

    return df


def interpolate(x: float, y: float, p: float) -> float:
    return y * p + (1 - p) * x


def generate_8000_cell_grid(grid_80):

    # Initialize an empty list to store subboxes
    subbox_list = []

    for index, row in grid_80.iterrows():
        alts = math.sin(row["Southern"] * math.pi / 180)
        altn = math.sin(row["Northern"] * math.pi / 180)

        for y in range(10):
            s = 180 * math.asin(interpolate(alts, altn, y * 0.1)) / math.pi
            n = 180 * math.asin(interpolate(alts, altn, (y + 1) * 0.1)) / math.pi
            for x in range(10):
                w = interpolate(row["Western"], row["Eastern"], x * 0.1)
                e = interpolate(row["Western"], row["Eastern"], (x + 1) * 0.1)

                # Create a DataFrame for the subbox
                subbox_df = pd.DataFrame(
                    {"Southern": [s], "Northern": [n], "Western": [w], "Eastern": [e]}
                )

                # Append the subbox DataFrame to the list
                subbox_list.append(subbox_df)

    # Concatenate all subboxes into a single DataFrame
    grid_8000 = pd.concat(subbox_list, ignore_index=True)

    # Calculate center coordinates for each box and add them as new columns
    center_coords = grid_8000.apply(calculate_center_coordinates, axis=1)
    grid_8000[["Center_Latitude", "Center_Longitude"]] = pd.DataFrame(
        center_coords.tolist(), index=grid_8000.index
    )

    # Calculate area of all 8000 cells
    grid_8000["Area"] = grid_8000.apply(calculate_area, axis=1)

    # Print the resulting DataFrame
    grid_8000 = grid_8000[["Center_Latitude", "Center_Longitude"]]
    grid_8000 = grid_8000.rename(
        columns={"Center_Latitude": "Latitude", "Center_Longitude": "Longitude"}
    )
    return grid_8000


"""
Step 2: Creation of 2x2 Grid

There are 16200 cells across the globe (90 lat x 180 lon).
Each cell's values are computed using station records within a 1200km radius.
    - Contributions are weighted according to distance to cell center
    (linearly decreasing to 0 at distance 1200km)
"""

# Standard library imports
from itertools import product

# 3rd party library imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports (tools functions)
from utilities import (
    calculate_distances,
    normalize_dict_values,
)


def create_grid() -> pd.DataFrame:
    """
    Create a grid of latitude and longitude values.

    This function generates a grid of latitude and longitude coordinates by using numpy's `np.arange` to create a range
    of values for both latitude and longitude. It then computes all possible combinations of these values and stores
    them in a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with two columns, 'Lat' and 'Lon', containing all possible combinations of latitude
        and longitude coordinates.
    """

    # Create latitude and longitude values using np.arange
    lat_values = np.arange(88.0, -90.0, -2.0, dtype=np.float32)
    lon_values = np.arange(0.0, 360.0, 2.0, dtype=np.float32)

    # Include coordinates for north/south poles
    polar_coordinates = [(90.0, 0.0), (-90.0, 0.0)]

    # Generate all possible combinations of latitude and longitude values
    combinations = list(product(lat_values, lon_values)) + polar_coordinates

    # Create a DataFrame from the combinations
    grid = pd.DataFrame(combinations, columns=["Latitude", "Longitude"])
    return grid


def collect_metadata() -> pd.DataFrame:
    """
    Collect station metadata from NASA GISS GISTEMP dataset.

    This function fetches station metadata from the NASA GISS GISTEMP dataset, specifically from the provided URL. The data
    is read as a fixed-width formatted (FWF) text file and stored in a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing station metadata, including columns for 'Station_ID', 'Latitude',
        'Longitude', 'Elevation', 'State', and 'Name'.
    """

    # Create station metadata dataframe
    meta_url = "https://data.giss.nasa.gov/pub/gistemp/v4.inv"
    column_widths = [11, 9, 10, 7, 3, 31]
    station_df: pd.DataFrame = pd.read_fwf(
        meta_url,
        widths=column_widths,
        header=None,
        names=["Station_ID", "Latitude", "Longitude", "Elevation", "State", "Name"],
    )
    return station_df


def find_nearby_stations(grid_df, station_df, distances, NEARBY_STATION_RADIUS):
    """
    Find nearby stations for each grid point based on specified distance radius.

    Parameters:
    - grid_df (pd.DataFrame): DataFrame containing grid coordinates with "Lat" and "Lon" columns.
    - station_df (pd.DataFrame): DataFrame containing station coordinates with "Latitude" and "Longitude" columns.
    - distances (np.ndarray): 2D array of distances between each grid point and station pair.
    - NEARBY_STATION_RADIUS (float): Maximum radius for considering stations as nearby.

    Returns:
    pd.DataFrame: Updated grid DataFrame with a new column "Nearby_Stations" containing dictionaries
                  mapping station IDs to their corresponding weights based on proximity.
    """
    nearby_dict_list = []

    distances[distances > NEARBY_STATION_RADIUS] = np.nan
    weights = 1.0 - (distances / NEARBY_STATION_RADIUS)

    with tqdm(
        range(len(grid_df)), desc="Finding nearby stations for each grid point"
    ) as progress_bar:

        for i in progress_bar:
            # Find indices of stations within the specified radius
            valid_indices = np.where(weights[i] <= 1.0)

            # Create a dictionary using numpy operations
            nearby_dict = {
                station_df.iloc[j]["Station_ID"]: weights[i, j]
                for j in valid_indices[0]
            }

            # Normalize weights to sum to 1
            nearby_dict = normalize_dict_values(nearby_dict)
            nearby_dict_list.append(nearby_dict)

            progress_bar.update(1)

        # Add the list of station IDs and weights as a new column
        grid_df["Nearby_Stations"] = nearby_dict_list
    return grid_df


########################################################################

EARTH_RADIUS = 6371
NEARBY_STATION_RADIUS = 1200

# Create equal area grid
grid_80 = generate_80_cell_grid()
grid_8000 = generate_8000_cell_grid(grid_80)

# Gather station metadata
station_df = collect_metadata()

# Create numpy array distances between all grid points / stations
distances = calculate_distances(grid_8000, station_df, EARTH_RADIUS)

# Add dictionary of station:weight pairs to grifd dataframe
grid_8000 = find_nearby_stations(
    grid_8000, station_df, distances, NEARBY_STATION_RADIUS
)

grid_8000.to_csv("step2_equal_area_output.csv")
