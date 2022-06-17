import logging
import re
from typing import List, Tuple, Set, Union

from ckonlpy.tag import Twitter
from flair.data import Tokenizer
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

logger = logging.getLogger(__name__)

AVAILABLE_TOKENIZERS = {'okt', 'komoran', 'mecab', 'hannanum', 'kkma'}
NEG_FILTER_POS = {
	"E",  # (Verbal endings)
	"J",  # (Ending Particle)
	"JKS",  # (Josa)
	"JKB",  # (Junction)
	"JX",  # (Junction)
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
		return list(map(lambda t: t[0], self.pos(text)))

	def __call__(self, text: str) -> List[str]:
		return self.tokenize(text)

	def pos(self, text: str) -> List[Tuple]:
		excluded_pos = self._excluded_pos
		pos_filtered = filter(lambda x: x[1] not in excluded_pos, self.word_tokenizer.pos(text))
		return list(pos_filtered)

	@property
	def name(self) -> str:
		return self.__class__.__name__ + "_" + self.tokenizer


VOCABS = [
	'내부', '제도', '신고서', '전자', '하이프사이클', '테스팅', '고객사', '커스터마이징', '테스터', '미연', '지구국', '메시지', '영상처리',
	'인터뷰', '조직도', '경영진'
]

DELETED_CHARS = "\\'|\\[|\\]|,|\\."


class TwitterTokenizer(Tokenizer):

	def __init__(self):
		self.pos_tagger = Twitter(use_twitter_dictionary=False)
		self.pos_tagger.add_dictionary(VOCABS, 'Noun')
		self.pos_tagger.add_a_template(('Modifier', 'Noun', 'Noun', 'Noun', 'Noun'))

	def tokenize(self, text: str) -> List[str]:
		return list(filter(lambda x: x, self.stcs_to_words(text).split(' ')))

	def __call__(self, text: str) -> List[str]:
		return self.tokenize(text)

	def stcs_to_words(self, text: str) -> str:
		stc_tagged = map(lambda t: t[0], self.pos_tagger.pos(text))
		stc = ' '.join(stc_tagged)
		stc_replaced = re.sub(DELETED_CHARS, u'', stc)
		return stc_replaced
