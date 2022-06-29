import logging
import math
from typing import Tuple, Literal, Dict, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
				converting_col = X_tr[col]
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
				from accutuning_helpers.text.embedder_tfidf import TfIdfTokenVectorizer
				vec = TfIdfTokenVectorizer(feature_name=col, **params)
			elif self._converter == 'BERT':
				## FIXME - flair version 에 맞게 huggingface version 4로 version up
				## FIXME - sentence transformer 기반 BERT vectorizer는 deprecated 될 예정
				from accutuning_helpers.text.embedder_bert import BERTVectorizer
				vec = BERTVectorizer(feature_name=col)
			self._vector_dict[col] = vec
			return vec
