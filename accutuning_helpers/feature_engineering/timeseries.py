import pandas as pd
# import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import dateutil.parser


def datetime_intervals(X, datetime_col):
    X.loc[:, datetime_col] = pd.to_datetime(X.loc[:, datetime_col], errors='coerce')
    X = X.sort_values(by=datetime_col)
    intervals = (X[datetime_col] - X[datetime_col].shift(1)).dropna()
    if intervals.nunique() == 1:
        fixed_interval = True
    else:
        fixed_interval = False
    return intervals, fixed_interval


# datetime이 있는 데이터를 일정한 간격으로 resample 합니다.
class AccutuningTimeseriesResample(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col, datetime_format=None, interval=None):
        self.datetime_col = datetime_col
        self.datetime_format = datetime_format
        self.interval = interval

    def fit(self, X, y=0, **fit_params):
        self.intervals, self.fixed_interval = datetime_intervals(X, self.datetime_col)
        return self

    def transform(self, X, y=0, **fit_params):
        target_column = X.loc[:, self.datetime_col]
        if self.datetime_format:
            X.loc[:, self.datetime_col] = pd.to_datetime(
                target_column,
                format=self.datetime_format,
                errors='coerce'
            )
        else:
            try:
                X.loc[:, self.datetime_col] = pd.to_datetime(
                    target_column,
                    infer_datetime_format=True
                )
            except Exception:
                def _parse(x):
                    try:
                        # TODO: 매번 format을 유추하기 때문에 느리다;
                        ret = dateutil.parser.parse(x)
                    except Exception:
                        ret = datetime.fromtimestamp(0)
                    return ret
                X.loc[:, self.datetime_col] = target_column.map(lambda x: _parse(x))
        X_tr = X.copy()
        X_tr.index = X_tr.loc[:, self.datetime_col]
        X_tr = X_tr.sort_index()
        if self.fixed_interval:
            pass
        elif self.interval is None:
            self.interval = self.intervals.mode()[0]
            X_tr = X_tr.resample(self.interval).mean()
        else:
            X_tr = X_tr.resample(self.interval).mean()
        return X_tr


# 일정한 간격으로 timeseries 데이터를 재배치하면서 결측값이 생기는 경우 Interpolate 합니다.
class AccutuningTimeseriesInterpolate(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear'):
        self.method = method

    def fit(self, X, y=0, **fit_params):
        if X.isnull().values.any():
            self.need_interpolation = True
        else:
            self.need_interpolation = False
        return self

    def transform(self, X, y=0, **fit_params):
        if self.need_interpolation:
            X_tr = X.interpolate(method=self.method)
        else:
            X_tr = X.copy()
        return X_tr


# 지정한 Window size와 Moving Average 방식대로 이동평균 컬럼을 추가합니다.
class AccutuningMovingAverage(BaseEstimator, TransformerMixin):
    """
    [(colname1, 5, 'ma'), (colname1, 10, 'ema'), (colname2, 5, 'ema')]
    """
    def __init__(self, how):
        self.how = how

    def fit(self, X, y=0, **fit_params):
        self.newcols_dict = {}

        for i in self.how:
            col, window, method = i[0], i[1], i[2]

            try:
                if method == 'ma':
                    newcol = X[col].rolling(window).mean()
                elif method == 'ema':
                    newcol = X[col].ewm(window).mean()

                colname = str(col) + '_' + str(window) + '_' + str(method)
                newcol.name = colname

                if col in self.newcols_dict.keys():
                    self.newcols_dict[col].append(newcol)
                else:
                    self.newcols_dict[col] = []
                    self.newcols_dict[col].append(newcol)

            except Exception as e:
                pass

        return self

    def transform(self, X, y=0, **fit_params):
        X_tr = X.copy()
        for i in self.newcols_dict.keys():
            target_idx = X.columns.get_loc(i)
            X_front = X_tr.columns[:target_idx + 1]
            X_back = X_tr.columns[target_idx + 1:]
            converted = pd.concat(self.newcols_dict[i], axis=1)
            X_tr = pd.concat([X_tr[X_front], converted, X_tr[X_back]], axis=1)
        X_tr = X_tr.dropna()
        return X_tr
