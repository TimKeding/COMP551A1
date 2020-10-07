import copy
import pandas as pd


class Data:

    def __init__(self):
        self.data = self.get_data()
        self.filter_out_zeros()

    def get_data(self):
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

        results = pd.merge(search, hosp, how="left", on=["open_covid_region_code", "date"])
        return results.fillna(0)

    def filter_out_zeros(self):
        return self.data.dropna(axis=1, how="all")

    def keep_15_symptoms(self):
        # sum all columns(symptoms)
        trim_data = copy.deepcopy(self.data)
        cropped = trim_data.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
        s = cropped.sum(axis=0)

        # get 15 symptoms with highest sum value(sum of popularity)
        arr = [0] * 15
        for j in range(15):
            # most common symptoms with sum of columns:
            # arr[j] = (s.idxmax(), ',', s.max())
            arr[j] = s.idxmax()
            s = s.drop(s.idxmax(), axis=0)

        # keep only 15 symptoms with highest popularity and convert to csv
        return trim_data.drop(
            columns=[
                col
                for col in trim_data.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
                if col not in arr
            ]
        )

    def merge_regions(self):
        return self.data.groupby(["date"]).sum().reset_index()
