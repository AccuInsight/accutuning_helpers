import os
from unittest import TestCase

import pandas as pd
import pickle

from accutuning_helpers.text.embedder_tfidf import TfIdfTokenVectorizer
from accutuning_helpers.text.tokenizer import KonlpyTokenizer


class TestTfidfTokenVectorizer(TestCase):

	def setUp(self) -> None:
		home = os.environ['ACCUTUNING_WORKSPACE']
		# datafile = os.path.join(home, 'data/naver_movie_comments_data_small.txt')
		# df = pd.read_csv(datafile, sep='\t', header=None, names=['movie_ids', 'comments', 'rates'])
		datafile = os.path.join(home, 'data/nnst_lt_10.csv')
		self.df = pd.read_csv(datafile)

		self.tokenizer = KonlpyTokenizer(tokenizer_name='mecab')
		# self.embedder = TfIdfTokenVectorizer(feature_name='comments', tokenizer=self.tokenizer)
		self.embedder = TfIdfTokenVectorizer(feature_name='stcs', tokenizer=self.tokenizer, min_df=0)

	def test_tokenize(self):
		assert len(self.df) > 0
		# comments = self.df['comments']
		comments = self.df['stcs']
		comment = comments[0]

		tokens = self.tokenizer(comment)
		print(f'tokens:{tokens}')
		# assert tokens == ['크리스토퍼', '놀란', '우리', '놀란', '다']

	def test_fit(self):
		X = self.df
		self.embedder.fit(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		assert self.embedder.embedding_length == 74
		vocab = self.embedder._vectorizer.vocabulary_
		keywords = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
		print(keywords[:5])
		# assert '힙' in keywords[:5]
		assert '해' in keywords[:5]

	def test_transform(self):
		X = self.df
		if self.embedder.embedding_length == 0:
			self.embedder.fit(X)

		vec = self.embedder.transform(X)
		# print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
		# assert vec.shape[1] == 9994
		assert vec.shape[1] == 76

	def test_fit_transform(self):
		X = self.df
		vec = self.embedder.fit_transform(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		assert self.embedder.embedding_length == 74
		print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')

	def test_save(self):
		with open('test_dump.pkl', 'wb') as f:
			pickle.dump(self.embedder, f)
			print(f'located: {f}')

	def test_load(self):
		with open('test_dump.pkl', 'rb') as f:
			embedder = pickle.load(f)
			print(embedder)
