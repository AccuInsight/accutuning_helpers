from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.dtypes.common import is_numeric_dtype
import logging

##########################################################################
# column 순서대로 제공되는 impute_strategies에 따라 결측값을 대체합니다.
# fit에서 train set으로부터 결측값을 대체할 imputing_values를 추출하고,
# transform에서 strategy에 해당하는 방법에 따라 결측값을 처리합니다. 
##########################################################################
class AccutuningNullImputerBycol(BaseEstimator, TransformerMixin):
    def __init__(self, impute_strategies):
        self.impute_strategies = impute_strategies

    def fit(self, X, y=0, **fit_params):
        self.strategies_dict = dict(zip(X.columns, self.impute_strategies))
        imputing_values = []
        for col in self.strategies_dict:
            try:
                imputing_col = X.loc[:, col]
            except KeyError:
                logging.critical(
                    'No such column name in the dataset - Null Imputer'
                )
            else:
                strategy = self.strategies_dict[col]
                val = None
                if strategy == 'MOST_FREQUENT':
                    val = imputing_col.mode()[0]
                elif strategy == 'UNKNOWN':
                    val = 'Unknown'
                elif is_numeric_dtype(imputing_col):
                    if strategy == 'MEAN':
                        val = imputing_col.mean()
                    elif strategy == 'MEDIAN':
                        val = imputing_col.median()
                    elif strategy == 'ZERO':
                        val = 0
                    elif strategy == 'MINIMUM':
                        val = imputing_col.min()
                imputing_values.append(val)
        
        self.imputing_dict = dict(zip(X.columns, imputing_values))
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        # impute_strategies는 column 순서대로 그에 해당하는 impute strategy list
        for col in self.strategies_dict:
            imputing_val = self.imputing_dict[col]
            strategy = self.strategies_dict[col]
            # numerical, categorical 공통 impute 방법
            if strategy in ('NONE', '', None):
                pass
            elif strategy == 'DROP':
                X_tr = X_tr.dropna(how='any', subset=[col], axis=0)
            # categorical impute 방법
            elif strategy == 'UNKNOWN':
                X_tr.loc[:, col] = X_tr.loc[:, col].fillna(value='Unknown')
            # numerical impute 방법
            elif strategy in ('MOST_FREQUENT', 'Unknown', 'MEAN', 'MEDIAN', 'ZERO', 'MINIMUM') and imputing_val is not None:
                X_tr.loc[:, col] = X_tr.loc[:, col].fillna(imputing_val)
        return X_tr
