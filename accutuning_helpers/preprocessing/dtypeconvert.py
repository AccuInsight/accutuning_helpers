from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

##########################################################################
# preprocessor 화면에서 받은 datatype_pair_match에 따라서 각 column들의 datatype을 변경합니다.
##########################################################################
class AccutuningDtypeConvert(BaseEstimator, TransformerMixin):
    def __init__(self, datatype_pair_match):
        self.datatype_pair_match = datatype_pair_match

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        for (col, typ) in self.datatype_pair_match:
            try:
                converting_col = X[col]
            except KeyError:
                logging.critical(
                    'No such column name in the dataset - dtype convert'
                )
            else:
                if typ == 'text':
                    from accutuning_helpers.feature_engineering.nlp import AccutuningVectorizer
                    vec = AccutuningVectorizer(feature_name=col)
                    X_tr = vec.fit_transform(X_tr)
                elif str(converting_col.dtype) != typ:
                    if typ == 'datetime64':
                        pass
                    elif typ == 'float64':
                        # to_numeric으로 float 변환 안되는 것들을 NaN으로 치환 후 변환
                        X_tr[col] = pd.to_numeric(X[col], errors='coerce').astype(typ)
                    else:
                        try:
                            X_tr[col] = X[col].astype(typ)
                        except ValueError:
                            logging.critical(
                                f'Failed to convert the datatype of column {col} to {str(typ)}. Set to default.'
                            )
        return X_tr
