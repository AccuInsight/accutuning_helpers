from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

##########################################################################
# CTGAN을 기반으로 데이터를 Augmentation 하여 새로운 데이터셋을 제공합니다.
# use_class_balancer일 경우 target_column의 각 class 별 row 수를 맞춥니다.
##########################################################################
class AccutuningSampler(BaseEstimator, TransformerMixin):
    def __init__(self, sample=0, epochs=5, discrete_columns=None, target_column=None, use_class_balancer = False):
        self.sample = sample
        self.epochs = epochs
        self.discrete_columns = discrete_columns
        self.target_column = target_column
        self.use_class_balancer = use_class_balancer
        self.max_row = 0
        self.ctgans = []

    def fit(self, X, y=0, **fit_params):
        from ctgan import CTGANSynthesizer
        if self.use_class_balancer:
            for cls in X[self.target_column].unique():
                ctgan = CTGANSynthesizer()
                new_X = X[(X[self.target_column] == cls)]
                if not self.sample and self.max_row < len(new_X):
                    self.max_row = len(new_X)
                ctgan.fit(new_X, self.discrete_columns, epochs=self.epochs)
                self.ctgans.append(ctgan)

        else:
            ctgan = CTGANSynthesizer()
            ctgan.fit(X, self.discrete_columns, epochs=self.epochs)
            self.ctgans.append(ctgan)

        return self

    def transform(self, X, y=0):
        if self.use_class_balancer:
            result = pd.DataFrame()
            if self.sample and self.sample > 0:
                class_num = len(self.ctgans)
                remainder = self.sample % class_num
                for ctgan in self.ctgans:
                    if remainder > 0:
                        tmp = ctgan.sample(int(self.sample / class_num) + 1)
                        remainder = remainder - 1
                    else:
                        tmp = ctgan.sample(int(self.sample / class_num))
                    result = result.append(tmp, ignore_index=True)
            else:
                for ctgan in self.ctgans:
                    tmp = ctgan.sample(self.max_row)
                    result = result.append(tmp, ignore_index=True)
            return result

        else:
            return self.ctgans[0].sample(self.sample)
