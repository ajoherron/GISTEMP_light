"""
Step 0: Downloading Data

Inputs include:
    - GHCN Temperature Data
    - GHCN Metadata
"""

# Standard library imports
import requests
from tqdm import tqdm

# 3rd-party library imports
import pandas as pd
import numpy as np


def get_GHCN_data(
    temp_url: str, meta_url: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """
    Retrieves and formats temperature data from the Global Historical Climatology Network (GHCN) dataset.

    Args:
    temp_url (str): The URL to the temperature data file in GHCN format.
    meta_url (str): The URL to the metadata file containing station information.

    Returns:
    df (pd.DataFrame): A Pandas DataFrame containing temperature data with station metadata.

    This function sends an HTTP GET request to the temperature data URL, processes the data to create
    a formatted DataFrame, replaces missing values with NaN, converts temperature values to degrees Celsius,
    and merges the data with station metadata based on station IDs. The resulting DataFrame includes
    columns for station latitude, longitude, and name, and is indexed by station IDs.
    """

    try:
        # Send an HTTP GET request to the URL
        response = requests.get(temp_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content of the response
            file_data: str = response.content.decode("utf-8")

            # Create a list to store formatted data
            formatted_data = []

            # Initialize tqdm with the total number of iterations
            total_iterations = len(file_data.split("\n"))
            progress_bar = tqdm(total=total_iterations, desc="Processing GHCN Data")

            # Loop through file data
            for line in file_data.split("\n"):
                # Update progress bar
                progress_bar.update(1)

                # Check if line is not empty
                if line.strip():
                    # Extract relevant data
                    # (Using code from GHCNV4Reader())
                    station_id = line[:11]
                    year = int(line[11:15])
                    values = [int(line[i : i + 5]) for i in range(19, 115, 8)]

                    # Append data to list
                    formatted_data.append([station_id, year] + values)

            # Close progress bar
            progress_bar.close()

            # Create DataFrame from formatted data
            column_names = ["Station_ID", "Year"] + [f"{i}" for i in range(1, 13)]
            df_GHCN = pd.DataFrame(formatted_data, columns=column_names)

            # Replace -9999 with NaN
            df_GHCN.replace(-9999, np.nan, inplace=True)

            # Convert temperature data to degrees Celsius
            month_columns = [f"{i}" for i in range(1, 13)]
            df_GHCN[month_columns] = df_GHCN[month_columns].divide(100)

            # Drop all years before start year & after end year
            start_year_mask = df_GHCN["Year"] >= start_year
            end_year_mask = df_GHCN["Year"] <= end_year
            df_GHCN = df_GHCN.loc[start_year_mask & end_year_mask]

        else:
            print("Failed to download the file. Status code: %s", response.status_code)

    except Exception as e:
        print("An error occurred: %s", e)

    # Pivot the dataframe on station ID
    pivoted_df = df_GHCN.pivot(index="Station_ID", columns="Year")

    # Flatten the multi-level columns and format them as desired
    pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]

    # Sort the columns by the month number
    sorted_columns = sorted(pivoted_df.columns, key=lambda x: int(x.split("_")[1]))

    # Reorder the dataframe columns
    pivoted_df = pivoted_df[sorted_columns]

    # Reset the index
    pivoted_df.reset_index(inplace=True)

    # Define the column widths, create meta data dataframe
    column_widths = [11, 9, 10, 7, 3, 31]
    df_meta = pd.read_fwf(
        meta_url,
        widths=column_widths,
        header=None,
        names=["Station_ID", "Latitude", "Longitude", "Elevation", "State", "Name"],
    )

    # Merge on station ID, set index, drop station names
    df = pd.merge(
        pivoted_df,
        df_meta[["Station_ID", "Latitude", "Longitude", "Name"]],
        on="Station_ID",
        how="left",
    )
    df.set_index("Station_ID", inplace=True)
    df.drop(columns="Name", inplace=True)

    return df


def step0(GHCN_TEMP_URL, GHCN_META_URL, START_YEAR, END_YEAR) -> pd.DataFrame:
    """
    Performs the initial data processing steps for the GHCN temperature dataset.

    Returns:
    df_GHCN (pd.DataFrame): A Pandas DataFrame containing filtered and formatted temperature data.

    This function retrieves temperature data from the Global Historical Climatology Network (GHCN) dataset,
    processes and formats the data, and returns a DataFrame. The data is first fetched using specified URLs,
    and is returned for further analysis.
    """
    df_GHCN = get_GHCN_data(GHCN_TEMP_URL, GHCN_META_URL, START_YEAR, END_YEAR)
    return df_GHCN
