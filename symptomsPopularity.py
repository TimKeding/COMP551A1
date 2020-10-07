import pandas as pd
import matplotlib.pyplot as plt


class SymptomPopularity:
    def __init__(self, data: pd.DataFrame):
        self.d = data

    def symptoms_popularity(self):
        self.d['hospitalized_new'] = self.d['hospitalized_new'] / 5
        df = self.d.loc[:, 'date':'symptom:Viral pneumonia']
        fig0, axs0 = plt.subplots(nrows=2, ncols=2)
        fig1, axs1 = plt.subplots(nrows=2, ncols=2)
        fig2, axs2 = plt.subplots(nrows=2, ncols=2)
        fig3, axs3 = plt.subplots(nrows=2, ncols=2)
        plt.ylim(0, 800)
        list_axes = [axs0, axs1, axs2, axs3]

        counter = 0

        for i, symptom in enumerate(df.loc[:, "symptom:Angular cheilitis":'symptom:Viral pneumonia']):
            axis = (list_axes[i // 4])[counter // 2, counter % 2]
            df.plot.bar(x='date', y=[symptom, ], ax=axis
                        , fontsize=5).set_ylabel("Symptom search popularity", fontsize=6)
            self.d.plot.bar(x='date', y='hospitalized_new', ax=axis,
                            color='r', alpha=0.5)
            axis.legend(loc='upper right', frameon=False, fontsize=6)
            counter += 1
            if counter > 3:
                counter = 0

        plt.show()
