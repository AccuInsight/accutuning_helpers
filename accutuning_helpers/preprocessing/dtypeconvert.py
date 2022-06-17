import logging
import math
from typing import Tuple, Literal, Dict, List, Union, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from flair.data import Sentence, Tokenizer
from flair.embeddings.document import DocumentEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from accutuning_helpers.text.embedder import TokenEmbedderBase
from accutuning_helpers.text.tokenizer import KonlpyTokenizer

# from accutuning_helpers.text.embedder_tfidf import TfIdfTokenVectorizer
from accutuning_helpers.text.embedder_bert import BERTVectorizer

logger = logging.getLogger(__name__)

TEXT_CONVERTER = Literal['tfidf', 'BERT', 'zero_shot', 'few_shot']


##########################################################################
# preprocessor 화면에서 받은 datatype_pair_match에 따라서 각 column들의 datatype을 변경합니다.
##########################################################################
class AccutuningDtypeConvert(BaseEstimator, TransformerMixin):

	def __init__(
			self,
			datatype_pair_match: List[Tuple[str, str]],
			text_converter: TEXT_CONVERTER = 'tfidf'
	):
		self.datatype_pair_match: Tuple[str, str] = datatype_pair_match
		self._converter = text_converter
		self._vector_dict: Dict[str, "TokenEmbedderBase"] = dict()

	def fit(self, X: pd.DataFrame, y=0, **fit_params) -> "AccutuningDtypeConvert":
		return self

	def transform(self, X: pd.DataFrame, y=0) -> pd.DataFrame:
		X_tr = X.copy()
		for (col, typ) in self.datatype_pair_match:
			try:
				converting_col = X[col]
			except KeyError:
				logger.critical(
					'No such column name in the dataset - dtype convert'
				)
			else:
				if typ == 'text':
					ratio = 0.001
					min_df = min(math.floor(len(converting_col) * ratio), 10)
					vec = self._vectorizer_factory(col, min_df=min_df)

					if vec.embedding_length == 0:  # not fitted
						X_tr = vec.fit_transform(X_tr, y=y)
					else:
						X_tr = vec.transform(X_tr)

				elif str(converting_col.dtype) != typ:
					if typ == 'datetime64':
						pass
					elif typ == 'float64':
						# to_numeric으로 float 변환 안되는 것들을 NaN으로 치환 후 변환
						X_tr[col] = pd.to_numeric(X[col], errors='coerce').astype(typ)
					else:
						try:
							X_tr[col] = X[col].astype(typ)
						except ValueError:
							logger.critical(
								f'Failed to convert the datatype of column {col} to {str(typ)}. Set to default.'
							)
		return X_tr

	def _vectorizer_factory(self, col: str, **params) -> "TokenEmbedderBase":
		if col in self._vector_dict:
			return self._vector_dict[col]
		else:  # register vectorizer
			if self._converter == 'tfidf':
# 				from accutuning_helpers.text.embedder_tfidf import TfIdfTokenVectorizer
				vec = TfIdfTokenVectorizer(feature_name=col, **params)
			elif self._converter == 'BERT':
				## FIXME - flair version 에 맞게 huggingface version 4로 version up
				## FIXME - sentence transformer 기반 BERT vectorizer는 deprecated 될 예정
# 				from accutuning_helpers.text.embedder_bert import BERTVectorizer
				vec = BERTVectorizer(feature_name=col)
			self._vector_dict[col] = vec
			return vec

		

class TfIdfTokenVectorizer(TokenEmbedderBase, DocumentEmbeddings):

	def __init__(
			self,
			feature_name: str,
			tokenizer: Tokenizer = KonlpyTokenizer('MeCab'),
			decode_error="replace",
			lowercase=False,
			min_df=10,  # configurable
			**vectorizer_params,
	):
		super(TfIdfTokenVectorizer, self).__init__(feature_name)
		vec = TfidfVectorizer(
			decode_error=decode_error,
			min_df=min_df,
			tokenizer=lambda x: tokenizer.tokenize(x),
			**vectorizer_params,
		)
		self._tokenizer = tokenizer
		self._vectorizer = vec
		self.name: str = "accutuning_text_tfidf"

	def fit(self, df: pd.DataFrame, y=None) -> "TfIdfTokenVectorizer":
		texts = df[self.feature_name].tolist()
		self._vectorizer.fit(texts, y=y)
		return self

	def vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
		# np.set_printoptions(threshold=100)
		corpus_embeddings = self.encode(df[self.feature_name].tolist())
		# columns = [
		# 	f'{self.feature_name}_{i}'
		# 	for i in range(corpus_embeddings.shape[1])
		# ]
		columns = self._vectorizer.get_feature_names()
		new_df = pd.DataFrame.sparse.from_spmatrix(corpus_embeddings, columns=columns)
		return new_df

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
