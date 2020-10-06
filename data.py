import pandas as pd
import numpy as np


def filter_out_zeros(df: pd.DataFrame):
    return df.dropna(axis=1, how="all")


def keep_15_symptoms(df: pd.DataFrame):
    # sum all columns(symptoms)
    cropped = df.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
    s = cropped.sum(axis=0)

    # get 15 symptoms with highest sum value(sum of popularity)
    arr = [0] * 15
    for j in range(15):
        # most common symptoms with sum of columns:
        # arr[j] = (s.idxmax(), ',', s.max())
        arr[j] = s.idxmax()
        s = s.drop(s.idxmax(), axis=0)

    # keep only 15 symptoms with highest popularity and convert to csv
    merged_file2 = df.drop(
        columns=[
            col
            for col in df.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
            if col not in arr
        ]
    )
    merged_file2.to_csv("merge2.csv")


def merge_regions(data: pd.DataFrame):
    return data.groupby(["date"]).sum().reset_index()


def get_data():
    hosp = pd.read_csv("hospitalization.csv")
    hosp = hosp[["open_covid_region_code", "date", "hospitalized_new"]]

    hosp["date"] = pd.to_datetime(hosp["date"], format="%Y-%m-%d") - pd.to_timedelta(
        7, unit="d"
    )
    hosp = (
        hosp.groupby(
            [
                "open_covid_region_code",
                pd.Grouper(key="date", freq="W-MON", closed="left"),
            ]
        )
        .sum()
        .reset_index()
    )

    search = pd.read_csv("2020_US_weekly_symptoms_dataset.csv")
    search["date"] = pd.to_datetime(search["date"], format="%Y-%m-%d")

    return pd.merge(search, hosp, how="left", on=["open_covid_region_code", "date"])
