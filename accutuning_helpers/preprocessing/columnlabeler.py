from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

##########################################################################
# Classification 문제에 해당할 경우 Target column을 int로 변환합니다.
##########################################################################
class AccutuningColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        self.le = None

    def fit(self, X, y=0, **fit_params):
        if self.column_name in X.columns:
            col = X.loc[:, self.column_name]
            try:
                task_type = type_of_target(col)
            except Exception:
                task_type = 'error'

            # if task_type == 'binary' or task_type.startswith('multiclass'):
            if task_type == 'binary' or col.dtype == 'object':
                self.le = LabelEncoder()
                self.le.fit(col)
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        if self.le and self.column_name in X.columns:
            X_tr.loc[:, self.column_name] = self.le.transform(X.loc[:, self.column_name])
        return X_tr
