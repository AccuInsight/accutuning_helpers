from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import time
import pandas as pd
import dateutil.parser


class AutoinsightDatetime64Converter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, datetime_format=None, populate_features=False, convert_timestamp=False):
        self.feature_name = feature_name
        self.datetime_format = datetime_format
        self.populate_features = populate_features
        self.convert_timestamp = convert_timestamp

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        cn = self.feature_name
        target_column = X.loc[:, cn]
        # X.loc[:, col] = pd.to_datetime(converting_col, format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
        if self.datetime_format:
            X.loc[:, cn] = pd.to_datetime(
                target_column,
                format=self.datetime_format,
                errors='coerce'
            )
            target_column = X.loc[:, cn]
        else:
            def _parse(x):
                try:
                    # TODO: 매번 format을 유추하기 때문에 느리다;
                    ret = dateutil.parser.parse(x)
                # except dateutil.parser.ParserError:
                #     ret = datetime.fromtimestamp(0)
                except Exception:
                    ret = datetime.fromtimestamp(0)
                return ret
            X.loc[:, cn] = target_column.map(lambda x: _parse(x))
            target_column = X.loc[:, cn]
        if self.populate_features:
            idx = X.columns.get_loc(cn)
            X.insert(idx + 1, cn + '_year', X[cn].dt.year)
            X.insert(idx + 2, cn + '_month', X[cn].dt.month)
            X.insert(idx + 3, cn + '_week', X[cn].dt.week)
            X.insert(idx + 4, cn + '_day', X[cn].dt.day)
            X.insert(idx + 5, cn + '_hour', X[cn].dt.hour)
            X.insert(idx + 6, cn + '_minute', X[cn].dt.minute)
            X.insert(idx + 7, cn + '_second', X[cn].dt.second)
            X.insert(idx + 8, cn + '_dayofweek', X[cn].dt.dayofweek)
            X.drop(columns=[cn], inplace=True)

        if self.convert_timestamp:
            X.loc[:, cn] = target_column.map(lambda x: time.mktime(
                x.timetuple()
            ))

        if not self.populate_features and not self.convert_timestamp:
            X.loc[:, cn] = target_column.astype('object')

        return X
