from typing import List, Union, Iterable

import numpy as np
import pandas as pd
from flair.data import Sentence, Tokenizer
from flair.embeddings.document import DocumentEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from accutuning_helpers.text.embedder import TokenEmbedderBase
from accutuning_helpers.text.tokenizer import KonlpyTokenizer


class TfIdfTokenVectorizer(TokenEmbedderBase, DocumentEmbeddings):

	def __init__(
			self,
			feature_name: str,
			tokenizer: Tokenizer = KonlpyTokenizer('MeCab'),
			decode_error="replace",
			lowercase=False,
			min_df=10,  # configurable
			ngram_range=(1,3),
			max_features=500,
			**vectorizer_params,
	):
		super(TfIdfTokenVectorizer, self).__init__(feature_name)
		vec = TfidfVectorizer(
			decode_error=decode_error,
			min_df=min_df,
			tokenizer=tokenizer,
			ngram_range=ngram_range,
			max_features=max_features,
			**vectorizer_params,
		)
		self._tokenizer = tokenizer
		self._vectorizer = vec
		self.name: str = "accutuning_text_tfidf"

	def fit(self, df: pd.DataFrame, y=None) -> "TfIdfTokenVectorizer":
		texts = df[self.feature_name].tolist()
		self._vectorizer.fit(texts, y=y)
		return self

	def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
		corpus_embeddings = self.encode(X[self.feature_name].tolist())
		corpus_columns = [ f"{self.feature_name}_{v}" for v in self._vectorizer.get_feature_names()]
		X = X.drop(self.feature_name, axis=1)

		## BUGFIX df.concat(X, corpus_df, axis=1) 이 NaN을 일으켜 numpy로 concat후 X부분만 형변환 함
		concated_ = np.append(X.to_numpy(), corpus_embeddings.A, axis=1)
		concated_df = pd.DataFrame(concated_, columns=list(X.columns) + corpus_columns)
		for c in X.columns:
			concated_df[c] = concated_df[c].astype(X[c].dtype)

		return concated_df

	def encode(self, sentences: Iterable[str]) -> np.ndarray:
		# documents = [self._tokenizer.tokenize(text) for text in X]
		# return self._vectorizer.transform(documents)
		return self._vectorizer.transform(sentences)

	@property
	def tokenizer(self) -> Tokenizer:
		return self._tokenizer

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
