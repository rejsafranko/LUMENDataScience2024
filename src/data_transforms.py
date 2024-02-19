import pandas as pd


# RULE 1
def handle_prereservation_arrival(df: pd.Series):
    df["datum_dolaska"] = pd.to_datetime(df["datum dolaska"])
    df["datum_kreiranja_rezervacije"] = pd.to_datetime(
        df["datum_kreiranja_rezervacije"]
    )
    return df[
        df["datum_dolaska"] >= df["datum_kreiranja_rezervacije"]
    ]  # Filter rows where arrival date is on or after reservation creation date.


# RULE 2
def handle_prearrival_checkout(df: pd.Series):
    df["datum_dolaska"] = pd.to_datetime(df["datum dolaska"])
    df["datum_odjave"] = pd.to_datetime(df["datum_odjave"])
    return df[df["datum_dolaska"] <= df["datum_odjave"]]


# RULE 3
def handle_prereservation_cancellation(df):
    df["datum_kreiranja_rezervacije"] = pd.to_datetime(
        df["datum_kreiranja_rezervacije"]
    )
    df["datum_otkazivanja_rezervacije"] = pd.to_datetime(
        df["datum_otkazivanja_rezervacije"]
    )
    condition = (df["datum_otkazivanja_rezervacije"].isnull()) | (
        df["datum_otkazivanja_rezervacije"] > df["datum_kreiranja_rezervacije"]
    )
    return df[condition]


# RULE 4


# case 1
def handle_overtime_cancellation_drop(df: pd.Series):
    df["datum_odjave"] = pd.to_datetime(df["datum_odjave"])
    df["datum_otkazivanja_rezervacije"] = pd.to_datetime(
        df["datum_otkazivanja_rezervacije"]
    )
    condition = (df["datum_otkazivanja_rezervacije"].isnull()) | (
        df["datum_otkazivanja_rezervacije"] < df["datum_odjave"]
    )
    return df[condition]


# case 2
def handle_overtime_cancellation_cast(df: pd.DataFrame):
    df["datum_otkazivanja_rezervacije"] = pd.to_datetime(
        df["datum_otkazivanja_rezervacije"]
    )
    df["datum_odjave"] = pd.to_datetime(df["datum_odjave"])
    condition = df["datum_otkazivanja_rezervacije"] > df["datum_odjave"]
    df.loc[condition, "status_rezervacije"] = "Check-Out"
    df.loc[condition, "datum_otkazivanja_rezervacije"] = None
    return df
