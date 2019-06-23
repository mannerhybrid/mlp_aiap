import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.tree import *
from sklearn.svm import *
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

class VoterPipeline:
    def __init__(self, df, target_idx=-1):
        self.estimators = {}
        self.top=3
        self.data = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        self.data.to_csv("../engineered_features.csv")
        self.x_df = self.data[self.data.columns[:target_idx]]
        self.y_df = self.data[self.data.columns[target_idx]]
        for i, model in enumerate([
            LinearRegression(),
            ARDRegression(),
            HuberRegressor(),
            PassiveAggressiveRegressor(),
            SGDRegressor(),
            RANSACRegressor(),
            TheilSenRegressor(),
            DummyRegressor(),
            AdaBoostRegressor(),
            BaggingRegressor(),
            GradientBoostingRegressor(),
            RandomForestRegressor(),
            MLPRegressor(),
            DecisionTreeRegressor(),
            LinearSVR(),
            SVR()
         ]):
            self.estimators["model_{}".format(i)] = model
        self.model_scores = {}
        self.best_model = {}

    

    def add_estimator(self, estimator):
        self.estimators = self.estimators.items().append("model_{}".format(len(self.estimators.items())), estimator)
    
    def delete_estimator(self, key):
        self.estimators.pop(key)

    def get_best_models(self):
        models = []
        i = 0
        kf = sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
        for k, model in self.estimators.items():
            scores= []
        #     print(str(model))
            for train_idx, test_idx in kf.split(self.x_df.values, self.y_df.values):
                x_train = self.x_df.loc[train_idx]
                y_train = self.y_df.loc[train_idx]
                x_test = self.x_df.loc[test_idx]
                y_test = self.y_df.loc[test_idx]
                mdl = model
                mdl.fit(x_train, y_train)
                predictions_test = mdl.predict(x_test)
                score = r2_score(predictions_test, y_test)
                scores.append(score)
                i += 1
            final_score = np.mean(scores)
            self.model_scores[k] = final_score
        best_scores = dict(sorted(self.model_scores.items(), key=lambda x:x[1], reverse=True)[:self.top])
        print(*[(self.estimators[k], self.model_scores[k]) for k in best_scores.keys()], sep="\n")
        self.models = [self.estimators[k] for k in best_scores.keys()]

    def plot_results(self):
        fig, ax = plt.subplots(3,self.top, figsize=(30,30))

        def z_score_normalize(arr):
            arr = (arr - arr.mean())/arr.std()
            return arr

        for i in range(self.top):
            mdl = self.models[i]
            x_train, x_test, y_train, y_test = train_test_split(
                                                    self.x_df.values,
                                                    self.y_df.values, 
                                                    test_size=0.33, 
                                                    random_state=42)
            predictions_train = mdl.predict(x_train)
            predictions_test = mdl.predict(x_test)
            targets_train = y_train
            targets_test = y_test
            sns.distplot(predictions_test, color="r", label="predictions", ax=ax[i][0])
            sns.distplot(targets_test, color="g", label="targets", ax=ax[i][0])
            ax[i][0].legend()
            ax[i][0].set_title("Test data")

            sns.distplot(predictions_train, color="r", label="predictions", ax=ax[i][1])
            sns.distplot(targets_train, color="g", label="targets", ax=ax[i][1])
            ax[i][1].legend()
            ax[i][1].set_title("Train data")

            ax[i][2].scatter(z_score_normalize(targets_train), z_score_normalize(predictions_train), label = 'train')
            ax[i][2].scatter(z_score_normalize(targets_test), z_score_normalize(predictions_test), label = 'test')
            ax[i][2].plot(np.linspace(-1,1, num=140), np.linspace(-1, 1, num=140), label='perfect predictions', c="green")
            ax[i][2].set_xlim((-2,2))
            ax[i][2].set_ylim((-2,2))
            ax[i][2].legend()
        plt.show()
