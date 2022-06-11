from abc import abstractmethod, ABC
from typing import List, Iterable, Union

import numpy as np
import pandas as pd
from flair.data import Tokenizer, Sentence
from sklearn.base import BaseEstimator, TransformerMixin


class TokenEmbedderBase(BaseEstimator, TransformerMixin, ABC):
	"""
	abstract class - 다양한 text embedder class를 추상화하는 class
	"""

	def __init__(self, feature_name: str):
		self.feature_name = feature_name

	@property
	@abstractmethod
	def tokenizer(self) -> Tokenizer:
		"""
			Embedder가 사용한 tokenizer를 반환. property 방식으로 접근
		Returns
		-------
			flair.data.Tokenizer
		"""
		pass

	def _add_embeddings_internal(self, sentences: List[Sentence]):
		pass

	def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
		vector_df = self.vectorize(X)
		X = X.drop(self.feature_name, axis=1)
		return pd.concat(
			[X, vector_df],
			axis=1
		)

	def vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
		np.set_printoptions(threshold=100)
		corpus_embeddings = self.encode(df[self.feature_name].tolist())
		if hasattr(corpus_embeddings, 'A'):
			corpus_embeddings = corpus_embeddings.A  # 변환 csr -> ndarray
		columns = [
			f'{self.feature_name}_{i}'
			for i in range(corpus_embeddings.shape[1])
		]
		return pd.DataFrame(
			corpus_embeddings,
			columns=columns
		)

	@abstractmethod
	def encode(self, sentences: Iterable[str]) -> Union[np.ndarray, List[np.ndarray]]:
		pass
