from unittest import TestCase

from accutuning_helpers.text.tokenizer import KonlpyTokenizer, TwitterTokenizer


class TestKonlpyTokenizer(TestCase):

	def test_tokenize(self):
		# text = '아버지가방에들어가신다'
		text = '크리스토퍼 놀란에게 [우리는] 놀란다'
		tok = KonlpyTokenizer(tokenizer_name='Mecab')
		tokens = tok.tokenize(text)
		print(tokens)
		assert len(tokens) == 4 # ['크리스토퍼', '놀란', '우리', '놀란다']

		pos = tok.pos(text)
		print(pos)
		assert list(map(lambda x: x[1],pos)) == ['NNP', 'NNP', 'NP', 'VV+EC']


class TestTwitterTokenizer(TestCase):

	def test_tokenize(self):
		# text = '아버지가방에들어가신다'
		text = '크리스토퍼 놀란에게 [우리는] 놀란다'
		tok = TwitterTokenizer()
		# tokens = tok.stcs_to_words(text)
		tokens = tok.tokenize(text)
		print(tokens)
		assert len(tokens) == 6 # ['크리스토퍼', '놀란', '에게', '우리', '는', '놀란다']
		# assert len(tokens) == 4 # ['크리스토퍼', '놀란', '우리', '놀란다']

		# pos = tok.pos(text)
		# print(pos)
		# assert list(map(lambda x: x[1],pos)) == ['NNP', 'NNP', 'NP', 'VV+EC']