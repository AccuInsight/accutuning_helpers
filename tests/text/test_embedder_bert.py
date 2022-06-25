import os
from unittest import TestCase

import pandas as pd

from accutuning_helpers.text.embedder_bert import BERTVectorizer


class TestBERTVectorizer(TestCase):

	def setUp(self) -> None:
		home = os.environ['ACCUTUNING_WORKSPACE']
		# datafile = os.path.join(home, 'data/naver_movie_comments_data_small.txt')
		# df = pd.read_csv(datafile, sep='\t', header=None, names=['movie_ids', 'comments', 'rates'])
		datafile = os.path.join(home, 'data/nnst_lt_10.csv')
		self.df = pd.read_csv(datafile)
		self.col_name = 'stcs'
		self.embedder = BERTVectorizer(feature_name=self.col_name)

	def test_tokenize(self):
		assert len(self.df) > 0
		comments = self.df[self.col_name]
		comment = comments[0]

		tokens = self.embedder.tokenizer.tokenize(comment)
		print(f'tokens:{tokens}')

	# assert tokens == ['크리스토퍼', '놀란', '우리', '놀란', '다']

	def test_fit(self):
		X = self.df
		self.embedder.fit(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		assert self.embedder.embedding_length == 512

	def test_transform(self):
		X = self.df
		vec = self.embedder.transform(X)
		# print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
		# assert vec.shape[1] == 9994
		assert vec.shape[1] == 512 + 2

	def test_fit_transform(self):
		X = self.df
		vec = self.embedder.fit_transform(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		assert self.embedder.embedding_length == 512
		print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
