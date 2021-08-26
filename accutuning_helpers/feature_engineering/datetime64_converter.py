from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import time
import pandas as pd
import dateutil.parser

##########################################################################
# Datetime을 표현하는 Column (feature_name)을 지정된 방식에 따라 변환하여 새 Column으로 추가합니다.
# datetime_format은 사용자가 입력한 컬럼의 표현 양식을 나타내며, 값이 입력되지 않아도 더 느리지만 추정하여 변환할 수 있습니다.
# populate_features일 경우 feature_name을 연, 월, 일, 시간, 분, 초 등의 새 Columns로 변환하여 추가합니다.
# convert_timestamp일 경우 feature_name을 정수형태의 timestamp로 변환한 column을 추가합니다. (ex.2015-07-31 -> 1438268400)
##########################################################################
class AccutuningDatetime64Converter(BaseEstimator, TransformerMixin):
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

        X_tr = X.copy()
        if self.populate_features:
            idx = X.columns.get_loc(cn)
            X_tr.insert(idx + 1, cn + '_year', X[cn].dt.year)
            X_tr.insert(idx + 2, cn + '_month', X[cn].dt.month)
            X_tr.insert(idx + 3, cn + '_week', X[cn].dt.week)
            X_tr.insert(idx + 4, cn + '_day', X[cn].dt.day)
            X_tr.insert(idx + 5, cn + '_hour', X[cn].dt.hour)
            X_tr.insert(idx + 6, cn + '_minute', X[cn].dt.minute)
            X_tr.insert(idx + 7, cn + '_second', X[cn].dt.second)
            X_tr.insert(idx + 8, cn + '_dayofweek', X[cn].dt.dayofweek)
            X_tr.drop(columns=[cn], inplace=True)

        if self.convert_timestamp:
            X_tr.loc[:, cn] = target_column.map(lambda x: time.mktime(
                x.timetuple()
            ))

        if not self.populate_features and not self.convert_timestamp:
            X_tr.loc[:, cn] = target_column.astype('object')

        return X_tr
