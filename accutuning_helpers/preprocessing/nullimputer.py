from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.dtypes.common import is_numeric_dtype
import logging

from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb 
import lightgbm as lgb
import pandas as pd


class Model_Imputer:
    """
    Parameters
    ----------
    classifier : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.
    regressor : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.
     n_iter : int
     Determines the number of iteration.
     initial_guess : string, callable or None, default='median'
     If ``mean``, the initial impuatation will use the median of the features.
     If ``median``, the initial impuatation will use the median of the features.
    """

    def __init__(self, method : str='MISSFOREST', initial_guess: str='median', n_iter: int=5):
        
        self.models = {
                "MISSFOREST" : [RandomForestClassifier(), RandomForestRegressor()],
                "KNN" : [KNeighborsClassifier(n_neighbors=3),KNeighborsRegressor(n_neighbors=3)], 
                "XGB" : [xgb.XGBClassifier(),xgb.XGBRegressor()], 
                "LGBM" : [lgb.LGBMClassifier(),lgb.LGBMRegressor()],  
            }
        self.classifier = self.models[method][0]
        self.regressor = self.models[method][1]
        self.initial_guess = initial_guess
        self.n_iter = n_iter

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.
        """
        miss_row = {}
        for c in X.columns:
            miss_row[c] = X[c][X[c].isnull() > 0].index
        miss_col = [k for k in miss_row.keys() if len(miss_row[k]) > 0]
        obs_row = X[X.isnull().sum(axis=1) == 0].index

        mappings = {}
        rev_mappings = {}
        for c in X.columns:
            if type(X[c].dropna().sample(n=1).values[0]) == str:
                mappings[c] = {k: v for k, v in zip(X[c].dropna().unique(), range(X[c].dropna().nunique()))}
                rev_mappings[c] = {v: k for k, v in zip(X[c].dropna().unique(), range(X[c].dropna().nunique()))}

        non_impute_cols = [c for c in X.columns if c not in mappings.keys()]
        # 1) Make an initial guess for all missing categorical/numeric values (e.g. mean, mode)
        for c in X.columns:
            # if datatype is numeric, fillna with mean or median
            if X[c].dtype in ['int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                              'float_','float16', 'float32', 'float64']:

                if self.initial_guess == 'mean':
                    X[c].fillna(X[c].mean(), inplace=True)

                else:
                    X[c].fillna(X[c].median(), inplace=True)

            # if datatype is categorical, fillna with mode
            else:
                X[c].fillna(X[c].mode().values[0], inplace=True)

        # Label Encoding
        for c in mappings:
            X[c].replace(mappings[c], inplace=True)
            X[c] = X[c].astype(int)

        iter = 0
        while True:
            for c in miss_col:
                if c in mappings:
                    estimator = self.classifier
                else:
                    estimator = self.regressor

                # Fit estimator with imputed X
                estimator.fit(X.drop(c, axis=1).loc[obs_row], X[c].loc[obs_row])

                # Predict the missing column with the trained estimator
                y_pred = estimator.predict(X.loc[miss_row[c]].drop(c, axis=1))
                y_pred = pd.Series(y_pred)
                y_pred.index = miss_row[c]
                # Update imputed matrix
                X.loc[miss_row[c], c] = y_pred
            # Check if Criteria is met
            if iter >= self.n_iter:
                break

            iter += 1
        # Reverse mapping
        for c in rev_mappings:
            X[c].replace(rev_mappings[c], inplace=True)

        self.X = X

        return X

##########################################################################
# column 순서대로 제공되는 impute_strategies에 따라 결측값을 대체합니다.
# fit에서 train set으로부터 결측값을 대체할 imputing_values를 추출하고,
# transform에서 strategy에 해당하는 방법에 따라 결측값을 처리합니다. 
##########################################################################
class AccutuningNullImputerBycol(BaseEstimator, TransformerMixin):
    def __init__(self, columns_name, impute_strategies):
        self.columns_name = columns_name
        self.impute_strategies = impute_strategies

    def fit(self, X, y=0, **fit_params):
        self.strategies_dict = dict(zip(self.columns_name, self.impute_strategies))
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
                elif strategy in ('MISSFOREST','KNN','XGB','LGBM'):
                    val = 'tbd'
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
        
        self.imputing_dict = dict(zip(self.columns_name, imputing_values))
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()

        if 'MISSFOREST' in self.strategies_dict.values() :
            imputer = Model_Imputer()
            MISSFOREST_imputed = imputer.fit_transform(X_tr)
        if 'KNN' in self.strategies_dict.values() :
            imputer = Model_Imputer(method = 'KNN')
            KNN_imputed = imputer.fit_transform(X_tr)
        if 'XGB' in self.strategies_dict.values() :
            imputer = Model_Imputer(method = 'XGB')
            XGB_imputed = imputer.fit_transform(X_tr)
        if 'LGBM' in self.strategies_dict.values() :
            imputer = Model_Imputer(method = 'LGBM')
            LGBM_imputed = imputer.fit_transform(X_tr)


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
            # model 기반 impute : categorical, numerical 둘 다 가능
            elif strategy == 'MISSFOREST' :
                X_tr.loc[:, col] = MISSFOREST_imputed.loc[:, col]
            elif strategy == 'KNN' :
                X_tr.loc[:, col] = KNN_imputed.loc[:, col]
            elif strategy == 'XGB' :
                X_tr.loc[:, col] = XGB_imputed.loc[:, col]
            elif strategy == 'LGBM' :
                X_tr.loc[:, col] = LGBM_imputed.loc[:, col]

        return X_tr
