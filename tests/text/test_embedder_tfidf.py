import os
import pickle
from unittest import TestCase

import pandas as pd

from accutuning_helpers.text.embedder_tfidf import TfIdfTokenVectorizer
from accutuning_helpers.text.tokenizer import KonlpyTokenizer


class TestTfidfTokenVectorizer(TestCase):

	def setUp(self) -> None:
		home = os.environ['ACCUTUNING_WORKSPACE']
		# datafile = os.path.join(home, 'data/naver_movie_comments_data_small.txt')
		# df = pd.read_csv(datafile, sep='\t', header=None, names=['movie_ids', 'comments', 'rates'])
		datafile = os.path.join(home, 'data/nnst_lt_1990.csv')
		# datafile = os.path.join(home, 'data/네이버영화평_sample.csv')
		df = pd.read_csv(datafile)
		self.train_df = df.iloc[:400]
		self.valid_df = df.iloc[400:]
		self.df = pd.concat([self.train_df, self.valid_df])

		self.tokenizer = KonlpyTokenizer(tokenizer_name='mecab')
		# self.embedder = TfIdfTokenVectorizer(feature_name='comments', tokenizer=self.tokenizer)
		self.embedder = TfIdfTokenVectorizer(
			feature_name='stcs',
			tokenizer=self.tokenizer,
			min_df=1,
			ngram_range=(1, 3),
			max_features=500
		)

	# self.embedder = TfIdfTokenVectorizer(feature_name='document', tokenizer=self.tokenizer, min_df=0)

	def test_tokenize(self):
		assert len(self.df) > 0
		# comments = self.df['comments']
		comments = self.df['stcs']
		# comments = self.df['document']
		comment = comments[0]

		tokens = self.tokenizer(comment)
		# assert tokens == ['크리스토퍼', '놀란', '우리', '놀란', '다']
		print(f'tokens:{tokens}')

	def test_fit(self):
		self.embedder.fit(self.df)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		# assert self.embedder.embedding_length == 74
		vocab = self.embedder._vectorizer.vocabulary_
		keywords = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
		print(keywords[:5])
		assert len(vocab.keys()) > 0

	def test_transform(self):
		if self.embedder.embedding_length == 0:
			self.embedder.fit(self.df)

		vec = self.embedder.transform(self.valid_df)
		print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
		# assert vec.shape[1] == 9994
		assert len(self.valid_df) == vec.shape[0]

	def test_fit_transform(self):
		X = self.df
		vec = self.embedder.fit_transform(X)
		print(f'\nvocab lenghth:{self.embedder.embedding_length}')
		# assert self.embedder.embedding_length == 9994
		# assert self.embedder.embedding_length == 74
		print(f'vector:{vec}')
		print(f'vector.shape:{vec.shape}')
		assert len(X) == vec.shape[0]

	def test_tfidf_max_features(self):
		if self.embedder.embedding_length == 0:
			self.embedder.fit(self.df)

		vec = self.embedder.transform(self.valid_df)
		assert self.embedder.embedding_length <= self.embedder._vectorizer.max_features

	def test_save(self):
		with open('test_dump.pkl', 'wb') as f:
			pickle.dump(self.embedder, f)
			print(f'located: {f}')

	def test_load(self):
		with open('test_dump.pkl', 'rb') as f:
			embedder = pickle.load(f)
			print(embedder)
