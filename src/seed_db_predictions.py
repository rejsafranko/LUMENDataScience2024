import os
import mysql.connector
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv("DB_HOST")  # RDS host
DATABASE = os.getenv("DB_NAME")
USER = os.getenv("MASTER_USERNAME")
PASSWORD = os.getenv("MASTER_PASSWORD")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def load_data_to_rds(csv_file, host, user, password, database):
    # Load data from CSV into a DataFrame.
    df = pd.read_csv(csv_file)

    # Connect to the AWS RDS MySQL database.
    connection = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = connection.cursor()

    # Create tables if they don't exist.
    for time in ["day", "week", "month"]: 
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lumendb.predictions_{time} (
                stay_date DATETIME,
                lower_bound INTEGER,
                upper_bound INTEGER,
                predicted_rooms INTEGER,
                total_rooms INTEGER
            )   
        """
        )

    # Insert data into the table.
    for time in ["day", "week", "month"]:    
        for index, row in df.iterrows():
            cursor.execute(
                f"""INSERT INTO lumendb.predictions_{time}
                (stay_date, lower_bound, upper_bound, predicted_rooms, total_rooms) 
                VALUES (STR_TO_DATE(%s, '%m/%d/%Y'), %s, %s, %s, %s)""",
                (
                    row["stay_date"],
                    row["lower_bound"],
                    row["upper_bound"],
                    row["predicted_rooms"],
                    row["total_rooms"],
                ),
            )

    connection.commit()

    # Close the cursor and connection.
    cursor.close()
    connection.close()


def main(args):
    load_data_to_rds(
        csv_file=args.dataset_path,
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
