import os
from unittest import TestCase

import pandas as pd

from accutuning_helpers.preprocessing.dtypeconvert import AccutuningDtypeConvert

HOME = os.environ['ACCUTUNING_WORKSPACE']


class TestAccutuningDtypeConvert(TestCase):

	def test_transform_nnst(self):
		datafile = os.path.join(HOME, 'data/nnst_lt_10.csv')
		df = pd.read_csv(datafile)
		converter = AccutuningDtypeConvert(
			datatype_pair_match=[('Unnamed: 0', 'int64'), ('stcs', 'text'), ('tags', 'object')]
		)
		ret_df = converter.transform(X=df)
		assert len(ret_df.columns) == 76 # Unnamed, tags, stcs_vocabulary 74

	def test_transform_naver(self):
		datafile = os.path.join(HOME, 'data/naver_movie_comments_data_small.txt')
		df = pd.read_csv(datafile, sep='\t', names=['movie_ids', 'comments', 'rates'])
		converter = AccutuningDtypeConvert(
			datatype_pair_match=[('movie_ids', 'int64'), ('comments', 'text'), ('rates', 'object')]
		)
		ret_df = converter.transform(X=df)
		assert len(ret_df.columns) == 9996 # movie_id, rates, tf_vocabulary 9994
