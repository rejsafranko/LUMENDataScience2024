import pandas as pd


def handle_overtime_cancellation(df: pd.DataFrame):
    df["datum_otkazivanja_rezervacije"] = pd.to_datetime(
        df["datum_otkazivanja_rezervacije"]
    )
    df["datum_odjave"] = pd.to_datetime(df["datum_odjave"])
    condition = df["datum_otkazivanja_rezervacije"] > df["datum_odjave"]
    df.loc[condition, "status_rezervacije"] = "Check-Out"
    df.loc[condition, "datum_otkazivanja_rezervacije"] = None
    return df
