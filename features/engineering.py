import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

class FeatureEngineer:
    def __init__(self, df, target_index=-1):
        self.data = df
        self.x_df = df[df.columns[:-1]]
        self.y_df = df[df.columns[-1]]
        self.translation_table = {}

    def add_poly(self, degree=3):
        self.data = self.x_df.join(self.y_df)
        self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df = self.data
        self.x_df = df[df.columns[:-1]]
        self.y_df = df[df.columns[-1]]
        print("Adding Polynomial Features \n")
        print(self.x_df.head())
        poly = sklearn.preprocessing.PolynomialFeatures(degree)
        print(self.x_df.columns)
        x = self.x_df.dropna(axis=1)
        print(x.columns)
        x_poly = poly.fit_transform(x)
        old_features = self.x_df.columns
        print(old_features)
        new_features = poly.get_feature_names()[1:len(old_features)+1]
        for f_old, f_new in zip(old_features, new_features):
            self.translation_table[f_new] = f_old
        print(self.translation_table)
        print(poly.get_feature_names())
        self.x_df = pd.DataFrame(x_poly, columns=poly.get_feature_names())

    def add_log(self, degree=3):
        x = self.x_df[self.x_df.columns[1:]]
        print("Adding Logarithmic Features \n")
        old_features = x.columns
        print(old_features)
        log_x = pd.DataFrame()
        for c in x.columns:
            log_x["log {}".format(c)] = x[c].apply(lambda x:np.log(x))
        
        new_features = ["log {}".format(var) for var in old_features]
        log_x = log_x.dropna(axis=1)
        print(log_x.head())
        print(new_features)
        self.x_df = x.join(log_x)
        print(x.shape)
        print(self.x_df.shape)

    def add_recip(self, degree=3):
        x = self.x_df
        print("Adding Reciprocal Features \n")
        recip_x = x[self.x_df.columns].apply(lambda x: 1/x)
        old_features = self.x_df.columns
        print(old_features)
        recip_x = pd.DataFrame()
        for c in x.columns:
            recip_x["1/{}".format(c)] = x[c].apply(lambda x:np.log(x))
        
        new_features = ["1/{}".format(var) for var in old_features]
        print(new_features)
        print(recip_x.head())
        recip_x = recip_x.dropna(axis=1)
        self.x_df = x.join(recip_x)
        print(x.shape)
        print(self.x_df.shape)

    def correlation_filter(self, limit=0.9):
        corr = self.x_df.corr()
        sns.heatmap(corr)
        plt.title("Correlation Matrix 2: Engineered Features")
        plt.show()
        initial_columns = self.x_df.columns
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    columns[j] = False  
        print(self.x_df.columns)
        selected_columns = self.x_df.columns[columns]
        self.x_df = self.x_df[selected_columns]
        print(self.x_df.columns)
        print("A total of {} columns have been removed.".format(len(initial_columns)- len(selected_columns)))

    def pvalue_filter(self, limit=0.05):
        numVars = len(self.x_df.columns)
        y = self.y_df
        x = self.x_df
        columns = np.full((numVars,), True, dtype=bool)
        for i in range(0, numVars):
            regOLS = sm.OLS(y, x.values).fit()
            maxVar = max(regOLS.pvalues)
            if maxVar > limit:
                for j in range(0, numVars - i):
                    if (regOLS.pvalues[j].astype(float) == maxVar):
                        columns[j] = False
        selected_columns = self.x_df.columns[columns]
        self.x_df = self.x_df[selected_columns]
        print(regOLS.summary())
        self.data = self.x_df.join(self.y_df)
        print("Selected features: {}".format(selected_columns))

        
            

        

    
