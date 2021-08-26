from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from .onehotencoder_support import OneHotEncoder as OneHotEncoder_

import numpy as np

##########################################################################
# 전체 categorical_features에 대하여 One Hot Encoding을 실시합니다.
# Advanced Settings의 One Hot Encoding을 True로 설정할 경우 사용됩니다.
# 사용자가 지정한 Column에 대해서만 OHE를 실시하는 AccutuningCategoryConverter와는 구분됩니다.
##########################################################################
class AccutuningOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, use_minimum_fraction=True, minimum_fraction=0.01,
                 categorical_features=None, random_state=None):

        self.use_minimum_fraction = use_minimum_fraction
        self.minimum_fraction = minimum_fraction
        self.categorical_features = categorical_features
        self.random_state = random_state

    def _fit(self, X, y=None):
        if self.use_minimum_fraction is False:
            self.minimum_fraction = None
        else:
            self.minimum_fraction = float(self.minimum_fraction)

        if self.categorical_features is not None:
            categorical_features = list(
                X.select_dtypes(include=['category', 'object']).columns
            )
        else:
            categorical_features = []

        self.preprocessor = OneHotEncoder_(
            minimum_fraction=self.minimum_fraction,
            categorical_features=categorical_features,
            sparse=True
        )

        return self.preprocessor.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        is_sparse = sparse.issparse(X)
        X = self._fit(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    def transform(self, X):
        is_sparse = sparse.issparse(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()


# # Imputation
# class Imputation(BaseEstimator, TransformerMixin):
#     def __init__(self, strategy='median', random_state=None):
#         self.strategy = strategy

#     def fit(self, X, y=None):
#         import sklearn.impute

#         self.preprocessor = sklearn.impute.SimpleImputer(
#             strategy=self.strategy, copy=False)
#         self.preprocessor = self.preprocessor.fit(X)
#         return self

#     def transform(self, X):
#         if self.preprocessor is None:
#             raise NotImplementedError()
#         return self.preprocessor.transform(X)


# # Variance Threshold
# class Variance_Threshold(BaseEstimator, TransformerMixin):
#     def __init__(self, random_state=None):
#         # VarianceThreshold does not support fit_transform (as of 0.19.1)!
#         pass

#     def fit(self, X, y=None):
#         self.preprocessor = VarianceThreshold(
#             threshold=0.0
#         )
#         self.preprocessor = self.preprocessor.fit(X)
#         return self

#     def transform(self, X):
#         if self.preprocessor is None:
#             raise NotImplementedError()
#         return self.preprocessor.transform(X)
