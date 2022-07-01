import logging
from typing import Iterable, List

import numpy as np
import pandas as pd
from flair.data import Tokenizer
from sentence_transformers import SentenceTransformer

from accutuning_helpers.text.embedder import TokenEmbedderBase

logger = logging.getLogger(__name__)


class BERTVectorizer(TokenEmbedderBase):

	def __init__(
			self,
			feature_name,
			bert_model_name='sentence-transformers/distiluse-base-multilingual-cased-v1',
			batch_size=16,
	):
		super(BERTVectorizer, self).__init__(feature_name)

		# 학습하지 않은 BERT 대신 sentence transformers가 자체적으로 multilingual로 학습한 vector로 수정
		# word_embedding_model = models.Transformer(bert_model_name)
		# pooling_model = models.Pooling(
		# 	word_embedding_model.get_word_embedding_dimension(),
		# 	pooling_mode_mean_tokens=True,
		# 	pooling_mode_cls_token=False,
		# 	pooling_mode_max_tokens=False)
		# self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
		self.model = SentenceTransformer(bert_model_name)
		self._batch_size = batch_size
		self._tokenizer = _Tokenizer(delegator=self.model)
		self._pool = self.model.start_multi_process_pool()
		logger.debug(f"SentenceTransformer start multi_process_pool")

	def __del__(self):
		self.model.stop_multi_process_pool(pool=self._pool)
		logger.debug('SentenceTransformer safely stop_multi_process_pool')

	def fit(self, X, y=0, **fit_params):
		return self

	def encode(self, sentences: Iterable[str]) -> List[np.ndarray]:
		# return self.model.encode(sentences, batch_size=self._batch_size)
		return self.model.encode_multi_process(
			[str(s) for s in sentences],
			pool=self._pool,
			batch_size=self._batch_size
		)

	@property
	def tokenizer(self) -> Tokenizer:
		return self._tokenizer

	@property
	def embedding_length(self) -> int:
		return self.model.get_sentence_embedding_dimension()


class _Tokenizer(Tokenizer):

	def __init__(self, delegator):
		self._delegator = delegator

	def tokenize(self, text: str) -> List[str]:
		return self._delegator.tokenize(text)


class AccutuningLabeler(BERTVectorizer):

	# def __init__(self, feature_name, classifier_fp, classifier_label_fp):
	def __init__(self, feature_name, classifier, classifier_labels, append_vectors=False):
		super().__init__(feature_name)

		self.classifier = classifier
		self.labels = classifier_labels
		self.append_vectors = append_vectors

	def transform(self, X, y=0):
		vector_df = self.vectorize(X)
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


