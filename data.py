import pandas as pd
import numpy as np


def filter_out_zeros(filename):
    file = pd.read_csv(filename)
    file2 = file.dropna(axis=1, how='all')
    file2.to_csv("merged_file2.csv")

def keep_15_symptoms(filename):
    # read new csv file with filtered 0s
    merged_file2 = pd.read_csv(filename)

    # sum all columns(symptoms)
    cropped = merged_file2.loc[:, 'symptom:Adrenal crisis':'symptom:Yawn']
    s = cropped.sum(axis=0)

    # get 15 symptoms with highest sum value(sum of popularity)
    arr = [0] * 15
    for j in range(15):
        # most common symptoms with sum of columns:
        # arr[j] = (s.idxmax(), ',', s.max())
        arr[j] = (s.idxmax())
        s = s.drop(s.idxmax(), axis=0)

    # keep only 15 symptoms with highest popularity and convert to csv
    merged_file2 = merged_file2.drop(
        columns=[col for col in merged_file2.loc[:, 'symptom:Adrenal crisis':'symptom:Yawn'] if col not in arr])
    merged_file2.to_csv("merge2.csv")


hosp = pd.read_csv("hospitalization.csv")
hosp = hosp[["open_covid_region_code", "date", "hospitalized_new"]]

hosp["date"] = pd.to_datetime(hosp["date"], format="%Y-%m-%d") - pd.to_timedelta(
    7, unit="d"
)
hosp = (
    hosp.groupby(
        ["open_covid_region_code", pd.Grouper(key="date", freq="W-MON", closed="left")]
    )
    .sum()
    .reset_index()
)

search = pd.read_csv("2020_US_weekly_symptoms_dataset.csv")
search["date"] = pd.to_datetime(search["date"], format="%Y-%m-%d")

merged = pd.merge(search, hosp, how="left", on=["open_covid_region_code", "date"])
merged.to_csv("merge.csv")

#function call - creates file merged_file2.csv
filter_out_zeros("merge.csv")
#function call- - creates file merge2.csv
keep_15_symptoms("merged_file2.csv")