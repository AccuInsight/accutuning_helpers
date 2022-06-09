from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator, TransformerMixin


class TokenEmbedderBase(BaseEstimator, TransformerMixin, ABC):
	"""
	abstract class - 다양한 text embedder class를 추상화하는 class
	"""
	@property
	@abstractmethod
	def get_tokenizer(self):
		pass
