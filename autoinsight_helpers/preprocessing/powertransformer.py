from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.dtypes.common import is_numeric_dtype
import logging
import numpy as np
import pandas as pd
from scipy import stats


class AutoinsightColTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, strategies):
        # self.trans_cols = trans_cols
        self.strategies = strategies

    def fit(self, X, y=0, **fit_params):
        new_strategies = []
        for (col, strategy) in self.strategies:
            try:
                converting_col = X.loc[:, col]
            except KeyError:
                logging.critical(
                    'No such column name in the dataset - power transformer'
                )
            else:
                if is_numeric_dtype(converting_col):
                    if strategy == 'LOG':
                        if any(x == 0 for x in converting_col.values):
                            logging.warning(
                                f'The column {converting_col.name} contains 0 which will be converted to -Inf. LOG Transformation is skipped.'
                            )
                            new_strategies.append('NONE')
                        elif any(x < 0 for x in converting_col.values):
                            logging.warning(
                                f'The column {converting_col.name} contains negative numbers which will be converted to nan. LOG Transformation is skipped.'
                            )
                            new_strategies.append('NONE')
                        else:
                            new_strategies.append('LOG')
                    elif strategy == 'SQUARED_ROOT':
                        if any(x < 0 for x in converting_col.values):
                            logging.warning(
                                f'The column {converting_col.name} contains negative numbers which will be converted to nan. SQUARED_ROOT Transformation is skipped'
                            )
                            new_strategies.append('NONE')
                        else:
                            new_strategies.append('SQUARED_ROOT')
                    elif strategy == 'SQUARE':
                        new_strategies.append('SQUARE')
                    # box-cox transformation, yeo-johnson transformation
                    elif strategy == 'BOX_COX_TRANSFORMATION':
                        if any(x <= 0 for x in converting_col.values):
                            logging.warning(
                                f'Data must be positive to use Box-Cox Transformation. Yeo-Johnson Transformation will be used instead for the column {converting_col.name}.'
                            )
                            new_strategies.append('YEO_JOHNSON_TRANSFORMATION')
                        else:
                            new_strategies.append('BOX_COX_TRANSFORMATION')
                    elif strategy == 'YEO_JOHNSON_TRANSFORMATION':
                        new_strategies.append('YEO_JOHNSON_TRANSFORMATION')
                    else:
                        logging.critical(
                            'Not a proper method'
                        )
                        new_strategies.append('NONE')
                else:
                    new_strategies.append('NONE')
        self.new_cols_strategies = zip(X.columns, new_strategies)
        return self

    def transform(self, X, y=0):
        
        X_tr = X.copy()
        for (col, strategy) in self.new_cols_strategies:
            if col not in X.columns:  # 존재하지않는 컬럼은 skip합니다
                continue
            
            converting_col = X.loc[:, col]
            if strategy == 'LOG':
                newcol = np.log(converting_col)
            elif strategy == 'SQUARED_ROOT':
                newcol = np.sqrt(converting_col)
            elif strategy == 'SQUARE':
                newcol = np.square(converting_col)
            # box-cox transformation, yeo-johnson transformation
            elif strategy == 'BOX_COX_TRANSFORMATION':
                try:
                    newcol, _ = stats.boxcox((converting_col + 1).astype(float))
                except ValueError:
                    # https://stackoverflow.com/questions/62116192/valueerror-data-must-not-be-constant
                    # prediction에서 row가 1개일때는 valueerror를 피할 수 없음.
                    newcol = None
            elif strategy == 'YEO_JOHNSON_TRANSFORMATION':
                newcol, _ = stats.yeojohnson(converting_col.astype(float))
            else:
                newcol = converting_col

            if newcol is not None:
                X_tr[col] = pd.Series(newcol)
        return X_tr
