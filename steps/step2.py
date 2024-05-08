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

            # Add dict to list
            nearby_dict_list.append(nearby_dict)
            progress_bar.update(1)

        # Add the list of station IDs and weights as a new column
        grid_df["Nearby_Stations"] = nearby_dict_list

    return grid_df


def step2(NEARBY_STATION_RADIUS, EARTH_RADIUS) -> pd.DataFrame:
    """
    This function represents Step 1 of the data processing pipeline. It involves the creation of a 2x2 grid of latitude
    and longitude values, gathering station metadata, and identifying nearby weather stations for each grid point along
    with their weights. The resulting information is stored in a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the grid points, station metadata, and nearby stations with their weights.
    """

    # Create 2x2 grid
    grid_df = create_grid()

    # Gather station metadata
    station_df = collect_metadata()

    # Create numpy array distances between all grid points / stations
    distances = calculate_distances(grid_df, station_df, EARTH_RADIUS)

    # Add dictionary of station:weight pairs to grid dataframe
    grid_df = find_nearby_stations(
        grid_df, station_df, distances, NEARBY_STATION_RADIUS
    )
    return grid_df
