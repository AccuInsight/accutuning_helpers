import numpy as np
from flair.data import Sentence, Tokenizer
from flair.embeddings.document import DocumentEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union, Iterable

from accutuning_helpers.text.embedder import TokenEmbedderBase


class TfIdfTokenEmbedder(TokenEmbedderBase, DocumentEmbeddings):

	def __init__(
			self,
			tokenizer: Tokenizer,
			decode_error="replace",
			lowercase=False,
			min_df=10,  # configurable
			**vectorizer_params,
	):
		super().__init__()
		vec = TfidfVectorizer(
			decode_error=decode_error,
			min_df=min_df,
			tokenizer=lambda x: tokenizer.tokenize(x),
			**vectorizer_params,
		)
		self._tokenizer = tokenizer
		self._vectorizer = vec
		self.name: str = "accutuning_text_tfidf"

	def fit(self, X: Iterable[str]) -> object:
		"""
		dataset X로부터 vocab, idf를 학습

		Parameters
		----------
		X : iterable

		y : None
			This parameter is not needed to compute tfidf.

		Returns
		-------
		self : object
			Fitted embedder.
		"""
		# documents = [self._tokenizer.tokenize(text) for text in X]
		# self._vectorizer.fit(documents)
		self._vectorizer.fit(X)
		return self

	def fit_transform(self, X, **fit_params):
		return super(TfIdfTokenEmbedder, self).fit_transform(X, **fit_params)

	def transform(self, X) -> np.ndarray:
		# documents = [self._tokenizer.tokenize(text) for text in X]
		# return self._vectorizer.transform(documents)
		return self._vectorizer.transform(X)

	@property
	def get_tokenizer(self) -> Tokenizer:
		self._tokenizer

	@property
	def embedding_length(self) -> int:
		vec = self._vectorizer
		return 0 if not hasattr(vec, 'vocabulary_') else len(vec.vocabulary_)

	def embed(self, sentences: Union[List[Sentence], Sentence]):
		"""Add embeddings to every sentence in the given list of sentences."""

		# if only one sentence is passed, convert to list of sentence
		if isinstance(sentences, Sentence):
			sentences = [sentences]

		import torch
		raw_sentences = [s.to_original_text() for s in sentences]
		tfidf_vectors = torch.from_numpy(self._vectorizer.transform(raw_sentences).A)

		for sentence_id, sentence in enumerate(sentences):
			sentence.set_embedding(self.name, tfidf_vectors[sentence_id])

	def _add_embeddings_internal(self, sentences: List[Sentence]):
		pass
