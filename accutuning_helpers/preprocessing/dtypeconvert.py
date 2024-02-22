import logging
import math
from typing import Dict, List, Literal, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

##########################################################################
# preprocessor 화면에서 받은 datatype_pair_match에 따라서 각 column들의 datatype을 변경합니다.
##########################################################################
class AccutuningDtypeConvert(BaseEstimator, TransformerMixin):
	def __init__(
			self,
			datatype_pair_match: List[Tuple[str, str]],
	):
		self.datatype_pair_match: Tuple[str, str] = datatype_pair_match
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
				if typ == 'text':  # TODO: text 타입을 없앤 후 제거
					pass
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
