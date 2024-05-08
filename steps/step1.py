"""
Step 1: Removal of bad data

Drop or adjust certain records (or parts of records).
This includes outliers / out of range reports.
Determined using configuration file (parameters/drop_rules.csv).
"""

# Standard library imports
import os
from tqdm import tqdm

# 3rd party imports
import pandas as pd
import numpy as np


def filter_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame based on latitude and longitude conditions.

    Args:
    df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.

    Returns:
    pd.DataFrame: The filtered DataFrame with rows where latitude is between -90 and 90,
    and longitude is between -180 and 180.
    """

    # Define latitude and longitude range conditions
    lat_condition = (df["Latitude"] >= -90) & (df["Latitude"] <= 90)
    lon_condition = (df["Longitude"] >= -180) & (df["Longitude"] <= 180)

    # Apply the conditions using the .loc indexer
    df_filtered = df.loc[lat_condition & lon_condition]

    # Calculate if there's faulty data, create logging message if so
    num_filtered = len(df) - len(df_filtered)
    if num_filtered > 0:
        print(f"Number of stations removed due to invalid coordinates: {num_filtered}")
    return df_filtered


def filter_by_rules(df) -> pd.DataFrame:
    """
    Filter a DataFrame based on specified rules provided in a CSV file.

    This function takes a DataFrame and filters it based on rules specified in a
    CSV file located at 'parameters/drop_rules.csv'. The rules in the CSV file
    indicate which data points to drop or set to NaN based on station IDs and
    year ranges.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to filter.

    Returns:
    pandas.DataFrame: A filtered DataFrame with data points removed according to drop_rules.csv.
    """

    # Initialize counter to keep track of dropped data points
    dropped_months = 0

    # Set path for drop rules file
    drop_rules_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "drop_rules.csv")
    )

    # Read in drop rules csv, create copy of input dataframe
    df_drop_rules = pd.read_csv(drop_rules_path, skipinitialspace=True)
    df_filtered = df.copy()

    # Initialize tqdm with the total number of iterations
    total_iterations = len(df_drop_rules)
    progress_bar = tqdm(total=total_iterations, desc="Removing faulty data")

    # Read through each row in drop rules dataframe
    for _, row in df_drop_rules.iterrows():
        # Update progress bar
        progress_bar.update(1)

        # Collect relevant information
        station = row["Station_ID"]
        year_range = row["Omit_Period"]
        if "-" in year_range:
            start_year = year_range.split("-")[0]
            end_year = year_range.split("-")[1]
            time_cols = [
                col
                for col in df_filtered.columns
                if not (col.startswith("Latitude") or col.startswith("Longitude"))
            ]
            if start_year == "0":
                # Dropping all station data (ex: 0-9999)
                if end_year == "9999":
                    df_filtered.loc[station, time_cols] = np.nan
                    dropped_months += len(time_cols)

                # Dropping all values before a given year (ex: 0-1950)
                else:
                    cols_to_keep = [
                        col
                        for col in time_cols
                        if int(col.split("_")[1]) > int(end_year)
                    ]
                    cols_to_replace = [
                        col for col in time_cols if col not in cols_to_keep
                    ]
                    df_filtered.loc[station, cols_to_replace] = np.nan
                    dropped_months += len(cols_to_replace)

            # Dropping all values after a given year (ex: 2012-9999)
            else:
                cols_to_keep = [
                    col for col in time_cols if int(col.split("_")[1]) < int(start_year)
                ]
                cols_to_replace = [col for col in time_cols if col not in cols_to_keep]
                df_filtered.loc[station, cols_to_replace] = np.nan
                dropped_months += len(cols_to_replace)

        # Dropping single months (ex: 2021/09)
        else:
            year = year_range.split("/")[0]
            month = year_range.split("/")[1]
            drop_col = str(int(month)) + "_" + str(year)
            df_filtered.loc[station, drop_col] = np.nan
            dropped_months += len(drop_col)

    # Close progress bar
    progress_bar.close()
    print(f"Number of data points removed: {dropped_months}")
    return df_filtered


def filter_by_stations(df, START_YEAR, END_YEAR):
    """
    Filter monthly timeseries for each month to ensure each includes at least 20 data points.
    Otherwise, set all values in the timeseries to NaN.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to filter.
    START_YEAR (int): The first year of the GISTEMP analysis
    END_YEAR (int): The last year of the GISTEMP analysis

    Returns:
    pandas.DataFrame: A DataFrame filtered according to the above rules.
    """

    # Initialize list for monthly anomaly dataframes
    monthly_dataframes = []

    # Loop through all months
    for month in range(1, 13):

        # Gather list of all columns for given month
        monthly_columns = []
        for year in range(START_YEAR, END_YEAR + 1):
            monthly_columns.append(str(month) + "_" + str(year))
        monthly_columns.append("Latitude")
        monthly_columns.append("Longitude")

        # Create monthly dataframe from column list
        df_monthly = df[monthly_columns]

        # Determine number of valid data points in baseline period
        month_columns = df_monthly.columns.drop(["Latitude", "Longitude"])
        timeseries_data = df_monthly[month_columns]
        valid_data_count = timeseries_data.count(axis=1)

        # Create column indicating if station has enough baseline data
        df_monthly = df_monthly.copy()
        df_monthly.loc[:, "Valid Data Count"] = valid_data_count
        df_monthly.loc[:, "Enough Monthly Data"] = df_monthly["Valid Data Count"] >= 20

        # Set rows without enough data to NaN
        columns_to_set_nan = df_monthly.columns[df_monthly.columns.str.contains("_")]
        df_monthly.loc[
            df_monthly["Enough Monthly Data"] == False, columns_to_set_nan
        ] = np.nan

        # Drop data count columns
        df_monthly = df_monthly.drop(
            columns=["Valid Data Count", "Enough Monthly Data"]
        )
        monthly_dataframes.append(df_monthly)

    # Combine filered monthly dataframes into total dataframe
    df_filtered = pd.concat(monthly_dataframes, axis=1)

    # Get list of sorted columns
    sorted_columns = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            column = str(month) + "_" + str(year)
            sorted_columns.append(column)
    sorted_columns.append("Latitude")
    sorted_columns.append("Longitude")

    # Reorder DataFrame columns
    df_filtered = df_filtered[sorted_columns]

    # Drop repeat Latitude / Longitude columns
    df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated()]
    return df_filtered


def step1(step0_output: pd.DataFrame, START_YEAR: int, END_YEAR: int) -> pd.DataFrame:
    """
    Applies data filtering and cleaning operations to the input DataFrame.

    Parameters:
        step0_output (pd.DataFrame): The initial DataFrame containing climate station data.

    Returns:
        pd.DataFrame: A cleaned and filtered DataFrame ready for further analysis.

    This function serves as a data processing step by applying two essential filtering operations:
    1. `filter_coordinates`: Filters the DataFrame based on geographical coordinates, retaining relevant stations.
    2. `filter_by_rules`: Filters the DataFrame based on exclusion rules, omitting specified stations and years.
    3. 'fitler_by_stations': Filters out station/month pairs that have fewer than 20 data points.

    The resulting DataFrame is cleaned of irrelevant stations and years according to specified rules
    and is ready for subsequent data analysis or visualization.
    """

    df_filtered_coords = filter_coordinates(step0_output)
    df_filtered_rules = filter_by_rules(df_filtered_coords)
    df_filtered_stations = filter_by_stations(df_filtered_rules, START_YEAR, END_YEAR)
    return df_filtered_stations
