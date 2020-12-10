from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import (
    models,
    # losses,
    SentenceTransformer,
)
import numpy as np
import pandas as pd
import pathlib
import pickle


class AutoinsightLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, classifier_fp, classifier_label_fp):
        self.feature_name = feature_name
        self.ohe = OneHotEncoder(sparse=False)
        self.classifier_fp = pathlib.Path(classifier_fp)
        self.classifier = pickle.loads(self.classifier_fp.read_bytes())
        self.classifier_label_fp = pathlib.Path(classifier_label_fp)
        self.labels = pickle.loads(self.classifier_label_fp.read_bytes())

        word_embedding_model = models.BERT('bert-base-multilingual-cased')
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def fit(self, X, y=0, **fit_params):
        return self

    def transform(self, X, y=0):
        np.set_printoptions(threshold=100)

        # logging.basicConfig(format='%(asctime)s - %(message)s',
        #                     datefmt='%Y-%m-%d %H:%M:%S',
        #                     level=logging.INFO,
        #                     handlers=[LoggingHandler()])
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
        vector_df = pd.DataFrame(
            corpus_embeddings,
            columns=columns
        )
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
        return pd.concat(
            [X, vector_df, tag_df],
            axis=1
        )
