from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.dtypes.common import is_numeric_dtype
import logging

##########################################################################
# outlier_cols에 해당되는 columns에 대하여 Outlier 제거를 실시합니다.
# BOX_PLOT_RULE, Z_SCORE 두 가지 방법을 선택할 수 있습니다.
# fitted는 기능이 전처리 과정에서만 작동하고, test set이 해당 클래스를 거치며 row를 지우는 일이 없도록 하는 역할을 수행합니다.
##########################################################################
class AccutuningOutlierBycol(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_cols, outlier_strategy, outlier_threshold=None, fitted=False):
        self.outlier_cols = outlier_cols
        self.outlier_strategy = outlier_strategy
        self.outlier_threshold = outlier_threshold
        self.fitted = fitted

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        if self.fitted:
            logging.critical(
                'AccutuningOutlierBycol: It is already transformed and will not be transformed again (for prediction).'
            )
        else:
            for col in self.outlier_cols:
                try:
                    converting_col = X.loc[:, col]
                except KeyError:
                    logging.critical(
                        'AccutuningOutlierBycol: No such column name in the dataset {}'.format(col)
                    )
                else:
                    if self.outlier_strategy == 'BOX_PLOT_RULE':
                        if is_numeric_dtype(converting_col):
                            q1 = converting_col.quantile(0.25)
                            q3 = converting_col.quantile(0.75)
                            IQR = q3 - q1
                            X_tr = X_tr[(X_tr[col] >= (q1 - 1.5 * IQR)) & (X_tr[col] <= (q3 + 1.5 * IQR))]
                            deleted = X_tr.shape[0] - sum((X_tr[col] >= (q1 - 1.5 * IQR)) & (X_tr[col] <= (q3 + 1.5 * IQR)))
                            logging.debug(f'AccutuningOutlierBycol: {deleted} rows deleted for column {col}')
                    elif self.outlier_strategy == 'Z_SCORE':
                        if is_numeric_dtype(converting_col):
                            if self.outlier_threshold is None:
                                self.outlier_threshold = 3
                            z = (X_tr[col] - converting_col.mean()) / converting_col.std()
                            X_tr = X_tr[abs(z) <= self.outlier_threshold]
                            deleted = sum(abs(z) > self.outlier_threshold)
                            logging.debug(f'AccutuningOutlierBycol: {deleted} rows deleted for column {col}')
                    else:
                        logging.critical(
                            'Not a proper method'
                        )
        self.fitted = True
        return X_tr
