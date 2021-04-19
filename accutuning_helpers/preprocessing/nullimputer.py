from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.dtypes.common import is_numeric_dtype
import logging


class AccutuningNullImputerBycol(BaseEstimator, TransformerMixin):
    def __init__(self, impute_strategies):
        self.impute_strategies = impute_strategies

    def fit(self, X, y=0, **fit_params):
        self.strategies_dict = dict(zip(X.columns, self.impute_strategies))
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        # impute_strategies는 column 순서대로 그에 해당하는 impute strategy list
        for col in self.strategies_dict:
            try:
                imputing_col = X.loc[:, col]
            except KeyError:
                logging.critical(
                    'No such column name in the dataset - Null Imputer'
                )
            else:
                strategy = self.strategies_dict[col]
                # numerical, categorical 공통 impute 방법
                if strategy in ('NONE', '', None):
                    pass
                elif strategy == 'DROP':
                    X_tr = X_tr.dropna(how='any', subset=[col], axis=0)
                elif strategy == 'MOST_FREQUENT':
                    X_tr.loc[:, col] = X_tr.loc[:, col].fillna(imputing_col.mode()[0])
                # categorical impute 방법
                elif strategy == 'UNKNOWN':
                    X_tr.loc[:, col] = X_tr.loc[:, col].fillna(value='Unknown')
                # numerical impute 방법
                elif is_numeric_dtype(imputing_col):
                    if strategy == 'MEAN':
                        i_mean = imputing_col.mean()
                        X_tr.loc[:, col] = X_tr.loc[:, col].fillna(value=i_mean)
                    elif strategy == 'MEDIAN':
                        i_median = imputing_col.median()
                        X_tr.loc[:, col] = X_tr.loc[:, col].fillna(value=i_median)
                    elif strategy == 'ZERO':
                        X_tr.loc[:, col] = X_tr.loc[:, col].fillna(value=0)
                    elif strategy == 'MINIMUM':
                        i_min = imputing_col.min()
                        X_tr.loc[:, col] = X_tr.loc[:, col].fillna(i_min)
        return X_tr
