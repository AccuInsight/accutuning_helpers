from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

##########################################################################
# 선택된 Column에 대하여 Lag Columns를 생성하여 추가합니다.
# lag만 제공될 경우 해당하는 단일 lag column을 생성하고, 
# lag, lag2 모두 제공될 경우 lag ~ lag2 범위에 해당하는 lag columns를 생성합니다.
# lag 값이 양수일 경우에는 컬럼 값을 아래로 내려 과거를 나타내고, 음수일 경우에는 위로 올려 미래를 나타냅니다. (pandas shift 참조)
##########################################################################
class AccutuningLagColumnAdder(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols=None, lag=None, lag2=None):
        self.target_cols = target_cols
        self.lag = lag
        self.lag2 = lag2

    def fit(self, X, y=0, **fit_params):
        self.row_size = X.shape[0]
        return self

    def transform(self, X, y=0, **fit_params):
        X_tr = X.copy()
        for target_col in self.target_cols:
            if self.lag2 is not None:
                lag_range = [
                    int(x) for x in range(self.lag, self.lag2 + 1)
                    if x != 0
                ]
                for i in lag_range:
                    col_name = '{}_lag{}'.format(target_col, i)

                    if target_col in X.columns:
                        if i < self.row_size:
                            tmp_col = pd.Series(X[target_col].shift(np.negative(i)))
                        else:
                            tmp_col = pd.Series(X[target_col]) # shift가 안 될 경우 그냥 해당 컬럼 추가 (predict할 때 컬럼 수 맞추기 위하여)

                        filler = 'ffill' if i > 0 else 'bfill'
                        tmp_col = tmp_col.fillna(method=filler)
                        X_tr.insert(
                            X.columns.get_loc(target_col),
                            col_name,
                            tmp_col
                        )
                    else:
                        X_tr[col_name] = np.NaN
            else:
                col_name = '{}_lag{}'.format(target_col, self.lag)
                if target_col in X.columns:
                    if self.lag < self.row_size:
                        tmp_col = pd.Series(X[target_col].shift(np.negative(self.lag)))
                    else:
                        tmp_col = pd.Series(X[target_col]) # shift가 안 될 경우 그냥 해당 컬럼 추가 (predict할 때 컬럼 수 맞추기 위하여)
                    filler = 'ffill' if self.lag > 0 else 'bfill'
                    tmp_col = tmp_col.fillna(method=filler)
                    X_tr.insert(
                        X.columns.get_loc(target_col),
                        col_name,
                        tmp_col
                    )
                else:
                    X_tr[col_name] = np.NaN
        return X_tr
