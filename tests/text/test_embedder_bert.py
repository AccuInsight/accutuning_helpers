import os
from time import perf_counter

import pandas as pd

from accutuning_helpers.text.embedder_bert import BERTVectorizer


def timer(fn):
	"""
	timer decorator
	"""

	def inner(*args, **kwargs):
		print(f'# Start {fn.__name__}')
		start_time = perf_counter()
		fn(*args, **kwargs)
		end_time = perf_counter()
		execution_time = end_time - start_time
		print(f'## {fn.__name__} took {execution_time:.5f}s to execute')

	return inner


class TestBERTVectorizer:

	def __init__(self):
		home = os.environ['ACCUTUNING_WORKSPACE']
		# datafile = os.path.join(home, 'data/naver_movie_comments_data_small.txt')
		# df = pd.read_csv(datafile, sep='\t', header=None, names=['movie_ids', 'comments', 'rates'])
		datafile = os.path.join(home, 'data/nnst_lt_1990.csv')
		self.df = pd.read_csv(datafile)
		self.col_name = 'stcs'
		self.embedder = BERTVectorizer(feature_name=self.col_name, batch_size=16)

	def test_tokenize(self):
		assert len(self.df) > 0
		comments = self.df[self.col_name]
		comment = comments[0]

		tokens = self.embedder.tokenizer.tokenize(comment)
		print(f'tokens:{tokens}')

	# assert tokens == ['크리스토퍼', '놀란', '우리', '놀란', '다']

	@timer
	def test_fit(self):
		X = self.df
		self.embedder.fit(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		assert self.embedder.embedding_length == 512

	@timer
	def test_transform(self):
		X = self.df
		vec = self.embedder.transform(X)
		# print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
		# assert vec.shape[1] == 9994
		assert vec.shape[1] == 512 + 2

	@timer
	def test_fit_transform(self):
		X = self.df
		vec = self.embedder.fit_transform(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		assert self.embedder.embedding_length == 512
		print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')


if __name__ == '__main__':
	test = TestBERTVectorizer()
	test.test_fit()
	test.test_transform()
	test.test_fit_transform()
