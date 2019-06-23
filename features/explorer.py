import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

class FeatureExplorer:
    def __init__(self, df):
        self.data = df

    def plot_correlogram(self):
        sns.pairplot(self.data)
        plt.show()
    
    def z_score_normalize(self, arr):
        arr = (arr - arr.mean())/arr.std()
        return arr

    def plot_feature_dists(self, target_index=-1):
        features = self.data.columns[:target_index]
        target = self.data.columns[target_index]
        num_features = len(features)
        fig, ax = plt.subplots(nrows=1 + (num_features//3), ncols=3, figsize=(45,45),sharex=True)
        blank_plots = (1 + num_features//3)*3 - num_features -1
        
        for num, i in enumerate(features):
            sns.distplot(self.z_score_normalize(self.data[i]), color="r", label=i, ax=ax[num//3][num % 3])
            sns.distplot(self.z_score_normalize(self.data[target]), color="g", label=target, ax=ax[num//3][num % 3])
            ax[num//3][num % 3].legend(loc="best")

        for i in range(blank_plots+1):
            print(i)
            n = num_features + i 
            print(n, n//3,n % 3)
            ax[n//3][n%3].axis("off")

        fig.suptitle("House Price Analysis")
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, wspace=0.15, hspace=0.2)
        plt.show()
    
    def correlation_matrix(self, include_target=True, target_index=-1):
        if include_target:
            corr = self.data[self.data.columns[:target_index]].corr()
        else:
            corr = self.data.corr()
        sns.heatmap(corr)
        plt.title("Correlation Matrix 1: Raw Features")
        plt.show()

        coordinates = np.where(abs(corr.values) > 0.9)
        correlations = list(map(lambda x:x[1], filter(lambda x:x[0] != x[1] and not len(self.data.columns)-1 in x, coordinates)))

        print("Features Highly Correlated to Target \n")
        for c in correlations:
            print(self.data.columns[c])
            self.data = self.data.drop(self.data.columns[c], axis = 1)