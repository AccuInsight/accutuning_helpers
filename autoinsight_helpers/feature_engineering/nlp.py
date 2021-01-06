from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import (
    models,
    # losses,
    SentenceTransformer,
)
import numpy as np
import pandas as pd
import pathlib
import pickle


class AutoinsightVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

        word_embedding_model = models.BERT('bert-base-multilingual-cased')
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def _vectorize(self, X):
        np.set_printoptions(threshold=100)
        corpus_embeddings = self.model.encode(
            [
                str(stc)
                for stc in X[self.feature_name].tolist()
            ]
        )
        columns = [
            f'{self.feature_name}_{i}'
            for i in range(corpus_embeddings.shape[1])
        ]
        return pd.DataFrame(
            corpus_embeddings,
            columns=columns
        )

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        vector_df = self._vectorize(X)
        X = X.drop(self.feature_name, axis=1)
        return pd.concat(
            [X, vector_df],
            axis=1
        )


class AutoinsightLabeler(AutoinsightVectorizer):
    # def __init__(self, feature_name, classifier_fp, classifier_label_fp):
    def __init__(self, feature_name, classifier, classifier_labels, append_vectors=False):
        super().__init__(feature_name)

        self.classifier = classifier
        self.labels = classifier_labels
        self.append_vectors = append_vectors

    def transform(self, X, y=0):
        vector_df = self._vectorize(X)
        tags = self.classifier.predict(vector_df)
        tags = [
            self.labels[int(tag)]
            for tag in tags
        ]
        tag_df = pd.DataFrame(
            tags,
            columns=[self.feature_name + '__tag']
        )
        X = X.drop(self.feature_name, axis=1)
        if self.append_vectors:
            return pd.concat(
                [X, vector_df, tag_df],
                axis=1
            )
        else:
            return pd.concat(
                [X, tag_df],
                axis=1
            )
