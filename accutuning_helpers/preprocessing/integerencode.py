from sklearn.base import BaseEstimator, TransformerMixin
from .ordinalencoder_tmp import OrdinalEncoder

##########################################################################
# DataFrame 내 Category나 String column일 경우 해당 column들은 Integer로 변환합니다.
# train set에서 없었던 값일 경우 transform 시에 unknown_value (기본값=-1)로 대체됩니다.
##########################################################################
class AccutuningIntegerEncode(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_value=-1):
        self.columns_to_encode = list()
        self.unknown_value = unknown_value
        self.oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=self.unknown_value)

    def fit(self, X, y=0, **fit_params):
        self.columns_to_encode = list(
            X.select_dtypes(include=['object'])
        )

        for col in self.columns_to_encode:
            # Null 값이 있으면 encoder에서 에러 발생한다. (추후 수정 필요)
            # integer(float) or 'NaN'이 섞여있어도 에러 발생한다.
            X.loc[:, col] = X.loc[:, col].fillna('NaN').apply(str)

        try:
            self.oe.fit(X.loc[:, self.columns_to_encode])
        except Exception as e:
            print(self.columns_to_encode)
            print(e)
            raise
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        for col in self.columns_to_encode:
            X_tr.loc[:, col] = X.loc[:, col].fillna('NaN').apply(str)
        X_tr.loc[:, self.columns_to_encode] = self.oe.transform(X_tr.loc[:, self.columns_to_encode]).astype('object')
        return X_tr
