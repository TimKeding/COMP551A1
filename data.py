import pandas as pd


def merge_regions(data: pd.DataFrame):
    return data.groupby(["date"]).sum().reset_index()


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

merge_regions(merged).to_csv("merge_region.csv")