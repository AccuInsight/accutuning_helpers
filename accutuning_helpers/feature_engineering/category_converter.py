from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class AccutuningCategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X, y=0, **fit_params):
        self.ohe.fit(X[[self.feature_name]], y)
        return self

    def transform(self, X, y=0):
        self.target_idx = X.columns.get_loc(self.feature_name)
        self.X_front = X.columns[:self.target_idx]
        self.X_back = X.columns[self.target_idx + 1:]
        new_X = self.ohe.transform(X[[self.feature_name]])
        new_df = pd.DataFrame(new_X, columns=[
            self.feature_name + '_' + str(cn)
            for cn in self.ohe.categories_[0]
        ])
        new_df.index = X.index
        X_tr = pd.concat([X[self.X_front], new_df, X[self.X_back]], axis=1)
        return X_tr
