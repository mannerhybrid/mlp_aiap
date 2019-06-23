from regression.voters import VoterPipeline
from features.engineering import FeatureEngineer
from features.explorer import FeatureExplorer
import pandas as pd
import requests


class StandardPipeline:
    def __init__(self, fname="real_estate.csv"):
        url = "https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv"
        r = requests.get(url, allow_redirects=True)
        open(fname, 'wb').write(r.content)
        self.pipe = []
        self.target_idx = -1
        self.start_df = pd.read_csv(fname)
        self.x_df = self.start_df[self.start_df.columns[:self.target_idx]]
        self.y_df = self.start_df[self.start_df.columns[self.target_idx]]

    def normal_pipeline(self):
        self.explorer = FeatureExplorer(self.start_df)
        self.explorer.plot_correlogram()
        self.explorer.plot_feature_dists()
        self.explorer.correlation_matrix()

        self.engineer = FeatureEngineer(self.explorer.data)
        self.engineer.add_recip()
        self.engineer.add_log()
        self.engineer.add_poly()
        self.engineer.correlation_filter()
        self.engineer.pvalue_filter()

        self.voter = VoterPipeline(self.engineer.data)
        self.voter.get_best_models()
        self.voter.plot_results()

        

def main():
    sp = StandardPipeline()
    sp.normal_pipeline()

if __name__ == "__main__":
    main()