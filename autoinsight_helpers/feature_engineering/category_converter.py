from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class AutoinsightCategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.ohe = OneHotEncoder(sparse=False)

    def fit(self, X, y=0, **fit_params):
        self.ohe.fit(X[[self.feature_name]], y)
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        new_X = self.ohe.transform(X[[self.feature_name]])
        new_df = pd.DataFrame(new_X, columns=[
            self.feature_name + '_' + str(cn)
            for cn in self.ohe.categories_[0]
        ])
        target_idx = X.columns.get_loc(self.feature_name)
        for idx, (cn, cd) in enumerate(new_df.iteritems()):
            X_tr.insert(target_idx + idx + 1, cn, cd)
        X_tr.drop(columns=[self.feature_name], inplace=True)
        return X_tr
