import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gdp_data_path", type=str, required=True, help="Path to GDP data."
    )
    parser.add_argument(
        "--hotel_data_path",
        type=str,
        required=True,
        help="Path to hotel occupancy data.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save data. Use data/util/",
    )
    return parser.parse_args()


def drop_non_reservation_years(df_hotel_occupancy: pd.DataFrame, df_gdp: pd.DataFrame):
    """
    Drops columns from df_gdp DataFrame that do not correspond to years present in reservation data.

    Parameters:
    - df_hotel_occupancy (pd.DataFrame): DataFrame containing reservation data.
    - df_gdp (pd.DataFrame): DataFrame containing GDP data.

    Returns:
    - pd.DataFrame: DataFrame df_gdp with non-reservation years columns dropped.
    """
    reservation_years = {
        year
        for year in pd.to_datetime(
            df_hotel_occupancy["datum_kreiranja_rezervacije"]
        ).dt.year
    }

    df_gdp = df_gdp.drop(
        list(
            set.symmetric_difference(
                set(map(str, reservation_years)), list(df_gdp.columns)[2:]
            )
        ),
        axis=1,
        errors="ignore",
    )

    return df_gdp


def format_dataframe_columns(df: pd.DataFrame):
    """
    Removes the "Country" column and renames the "Country Code" column to "code" in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be modified.

    Returns:
    - pd.DataFrame: The modified DataFrame with the specified column operations.
    """
    df.drop(["Country "], axis=1, inplace=True)
    df.rename(columns={"Country Code": "code"}, inplace=True)
    return df


def main(args):
    df_hotel_occupancy = pd.read_parquet(args.hotel_data_path)
    df_gdp = pd.read_csv(args.gdp_data_path)
    df_gdp = drop_non_reservation_years(df_hotel_occupancy, df_gdp)
    df_gdp = format_dataframe_columns(df_gdp)
    df_gdp.to_csv(args.save_path + "prepared_GDP.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
