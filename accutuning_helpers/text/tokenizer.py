import logging
from typing import List, Tuple, Set, Union

from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

from flair.data import Tokenizer

logger = logging.getLogger(__name__)

AVAILABLE_TOKENIZERS = {'okt', 'komoran', 'mecab', 'hannanum', 'kkma'}
NEG_FILTER_POS = {
	"E",  # (Verbal endings)
	"J",  # (Ending Particle)
	"JKS",  # (Josa)
	"JKB",  # (Junction)
	"JX", 	# (Junction)
	"MM",  # (Modifier)
	"SP",  # (Space)
	"SSC",  # (Closing brackets)
	"SSO",  # (Opening brackets)
	"SC",  # (Separator)
	"SE",  # (Ellipsis)
	"XPN",  # (Prefix)
	"XSA",  # (Adjective Suffix)
	"XSN",  # (Noun Suffix)
	"XSV",  # (Verb Suffix)
	"UNA",  # (Unknown)
	"NA",  # (Unknown)
	"VSV",  # (Unknown)
	"VCN",  # (Negative designator)
	"VCP"  # (Positive designator)
}

# 임시로 지정


class KonlpyTokenizer(Tokenizer):
	"""
	Tokenizer using koNLP, a third party library which supports
	multiple Korean tokenizer such as MeCab, Komoran, okt, hannanum, kkma.

	default - NEG_FILTER_POS 에 속한 품사는 제외함

	For further details see:
		https://github.com/konlpy/konlpy
	"""

	def __init__(
			self,
			tokenizer_name: str,
			excluded_pos: Union[List, Set] = NEG_FILTER_POS,
	):
		super(KonlpyTokenizer, self).__init__()

		name = tokenizer_name.lower()

		if name not in AVAILABLE_TOKENIZERS:
			raise NotImplementedError(
				f"Currently, {tokenizer_name} is only supported. Supported tokenizers: {AVAILABLE_TOKENIZERS}."
			)

		if name == 'okt':
			tokenizer_ = Okt()
		elif name == 'komoran':
			tokenizer_ = Komoran()
		elif name == 'mecab':
			tokenizer_ = Mecab()
		elif name == 'hannanum':
			tokenizer_ = Hannanum()
		elif name == 'kkma':
			tokenizer_ = Kkma()

		self.tokenizer_name = tokenizer_name
		self.word_tokenizer = tokenizer_
		self._excluded_pos = excluded_pos or NEG_FILTER_POS

	def tokenize(self, text: str) -> List[str]:
		pos_filtered = map(lambda t: t[0], self.pos(text))
		return list(pos_filtered)

	def pos(self, text: str) -> List[Tuple]:
		excluded_pos = self._excluded_pos
		pos_filtered = filter(lambda x: x[1] not in excluded_pos, self.word_tokenizer.pos(text))
		return list(pos_filtered)

	@property
	def name(self) -> str:
		return self.__class__.__name__ + "_" + self.tokenizer


if __name__ == "__main__":
	# text = '아버지가방에들어가신다'
	text = '크리스토퍼 놀란에게 우리는 놀란다'
	tok = KonlpyTokenizer(tokenizer_name='Mecab')
	tokens = tok.tokenize(text)
	print(tokens)
	pos = tok.pos(text)
	print(pos)
