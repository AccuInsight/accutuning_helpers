from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging


class AutoinsightDtypeConvert(BaseEstimator, TransformerMixin):
    # preprocessor 화면에서 Type 받아서 변환하는 class

    def __init__(self, datatype_pair_match):
        self.datatype_pair_match = datatype_pair_match

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        for (col, typ) in self.datatype_pair_match:
            try:
                converting_col = X[col]
            except KeyError:
                logging.critical(
                    'No such column name in the dataset - dtype convert'
                )
            else:
                if str(converting_col.dtype) != typ:
                    if typ == 'datetime64':
                        pass
                    elif typ == 'float64':
                        # to_numeric으로 float 변환 안되는 것들을 NaN으로 치환 후 변환
                        X[col] = pd.to_numeric(X[col], errors='coerce').astype(typ)
                    else:
                        try:
                            X[col] = X[col].astype(typ)
                        except ValueError:
                            logging.critical(
                                f'Failed to convert the datatype of column {col}.'
                            )
                            raise
        return X
