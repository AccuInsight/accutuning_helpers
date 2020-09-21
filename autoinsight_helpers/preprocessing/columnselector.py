from sklearn.base import BaseEstimator, TransformerMixin


class AutoinsightColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        inter = set(X.columns) & set(self.column_names)
        filtered = [
            c
            for c in self.column_names
            if c in inter
        ]  # 순서보존을 위해 inter사용대신 재작업
        return X.loc[:, filtered]
