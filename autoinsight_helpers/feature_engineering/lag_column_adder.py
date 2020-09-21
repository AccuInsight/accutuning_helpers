from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class AutoinsightLagColumnAdder(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols=None, lag=None, lag2=None):
        self.target_cols = target_cols
        self.lag = lag
        self.lag2 = lag2

    def fit(self, X, y=0, **fit_params):
        self.row_size = X.shape[0]
        return self

    def transform(self, X, y=0, **fit_params):
        for target_col in self.target_cols:
            if self.lag2:
                lag_range = [
                    int(x) for x in range(self.lag, self.lag2 + 1)
                    if x != 0
                ]
                for i in lag_range:
                    if i < self.row_size:
                        tmp_col = pd.Series(X[target_col].shift(i))
                    else:
                        tmp_col = pd.Series(X[target_col]) # shift가 안 될 경우 그냥 해당 컬럼 추가 (predict할 때 컬럼 수 맞추기 위하여)

                    filler = 'ffill' if i < 0 else 'bfill'
                    tmp_col = tmp_col.fillna(method=filler)
                    col_name = '{}_lag{}'.format(target_col, i)
                    X.insert(
                        X.columns.get_loc(target_col),
                        col_name,
                        tmp_col
                    )
            else:
                if self.lag < self.row_size:
                    tmp_col = pd.Series(X[target_col].shift(self.lag))
                else:
                    tmp_col = pd.Series(X[target_col]) # shift가 안 될 경우 그냥 해당 컬럼 추가 (predict할 때 컬럼 수 맞추기 위하여)
                filler = 'ffill' if self.lag < 0 else 'bfill'
                tmp_col = tmp_col.fillna(method=filler)
                col_name = '{}_lag{}'.format(target_col, self.lag)
                X.insert(
                    X.columns.get_loc(target_col),
                    col_name,
                    tmp_col
                )
        return X
