import logging
from typing import Dict, List, Literal

import datasets
from accutuning_helpers.text.meta_learning import MetaLearner
from flair.data import Corpus, Sentence, Tokenizer
from flair.datasets.document_classification import FlairDataset
from flair.models import TARSClassifier
from flair.tokenization import SegtokTokenizer
from flair.trainers import ModelTrainer
from torch.optim import AdamW

logger = logging.getLogger()


def fetch(
		cls,
		**corpus_args
) -> Corpus:
	"""
	다루고 있는 한글 Dataset 종류
	1. KLUE benchmark (https://klue-benchmark.com/tasks) 중 - ynat, nli, sts
	-  Korean HateSpeech Dataset (https://github.com/kocohub/korean-hate-speech)
	"""

	# 1. fetch from huggingface
	dset_ = datasets.load_dataset(*cls.dataset_args)
	train_, dev_, test_ = dset_.get('train'), dset_.get('validation'), dset_.get('test')
	tokenizer = SegtokTokenizer()

	# 2. convert
	_train, _dev, _test = None, None, None
	if train_:
		_train = HuggingfaceDataset.__new__(cls)
		_train.__init__(dataset=train_, tokenizer=tokenizer)
	if dev_:
		_dev = HuggingfaceDataset.__new__(cls)
		_dev.__init__(dataset=dev_, tokenizer=tokenizer)
	if test_:
		_test = HuggingfaceDataset.__new__(cls)
		_test.__init__(dataset=test_, tokenizer=tokenizer)

	# 3. bag in the corpus
	return Corpus(train=_train, dev=_dev, test=_test, name=corpus_args.get('name', _train.task_name), **corpus_args)


class HuggingfaceDataset(FlairDataset):
	def is_in_memory(self) -> bool:
		return True


class KlueYnatDataset(HuggingfaceDataset):
	dataset_args = ['klue', 'ynat']
	task_name = 'klue-ynat'
	label_name_map: Dict[int, str] = {
		0: "IT 과학",
		1: "경제",
		2: "사회",
		3: "생활 문화",
		4: "세계",
		5: "스포츠",
		6: "정치"
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		texts = dataset.data['title']  # features - [guid, title, label, url, date]
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self._sentences: List[Sentence] = [
			Sentence(str(text), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for text, tag in zip(texts, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self._sentences[index]

	def __len__(self) -> int:
		return len(self._sentences)


class KlueNliDataset(HuggingfaceDataset):
	dataset_args = ['klue', 'nli']
	task_name = 'klue-nli'
	label_name_map: Dict[int, str] = {
		0: "유추 가능",
		1: "무관함",
		2: "모순",
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		self.sep = " [SEP] "

		# features - [guid, source, premise, hypothesis, label]
		prems, hypos = dataset.data['premise'], dataset.data['hypothesis']
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self.pairs: List[Sentence] = [
			Sentence(text=str(prem) + self.sep + str(hypo), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for prem, hypo, tag in zip(prems, hypos, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self.pairs[index]

	def __len__(self) -> int:
		return len(self.pairs)


class KlueStsDataset(HuggingfaceDataset):
	dataset_args = ['klue', 'sts']
	task_name = 'klue-sts'
	label_name_map: Dict[int, str] = {
		0: "다른 의미",
		1: "같은 의미",
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		self.sep = " [SEP] "

		# features - [guid (string)	source (string)	sentence1 (string)	sentence2 (string)	labels (json)]
		# klue-sts-v1_train_00000
		# airbnb-rtt
		# 숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.
		# 숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.
		# { "label": 3.7, "real-label": 3.714285714285714, "binary-label": 1 }
		s1s, s2s = dataset.data['sentence1'], dataset.data['sentence2']
		tags = [self.label_name_map[d['binary-label'].as_py()] for d in dataset.data['labels']]
		self.pairs: List[Sentence] = [
			Sentence(text=str(s1) + self.sep + str(s2), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for s1, s2, tag in zip(s1s, s2s, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self.pairs[index]

	def __len__(self) -> int:
		return len(self.pairs)


class PawsXDataset(HuggingfaceDataset):
	dataset_args = ['PAWS-X', 'ko']
	task_name = 'paws-x'
	label_name_map: Dict[int, str] = {
		0: "전혀 다른 의미의 문장",
		1: "동일한 의미의 문장",
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		self.sep = " [SEP] "

		# features - [id, sentence1, sentence2, label]
		s1s, s2s = dataset.data['sentence1'], dataset.data['sentence2']
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self.pairs: List[Sentence] = [
			Sentence(text=str(s1) + self.sep + str(s2), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for s1, s2, tag in zip(s1s, s2s, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self.pairs[index]

	def __len__(self) -> int:
		return len(self.pairs)


class NaverSentimentMovieCommentsDataset(HuggingfaceDataset):
	dataset_args = ['nsmc']
	task_name = 'nsmc'
	label_name_map: Dict[int, str] = {
		0: "부정적인 평가",
		1: "긍정적인 평가",
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		# features - [id, document, label]
		texts = dataset.data['document']
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self._sentences: List[Sentence] = [
			Sentence(str(text), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for text, tag in zip(texts, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self._sentences[index]

	def __len__(self) -> int:
		return len(self._sentences)


class KoreanRestaurantReviewsDataset(HuggingfaceDataset):
	dataset_args = ["Wittgensteinian/KR3"]
	task_name = 'kr3'
	label_name_map: Dict[int, str] = {
		0: "부정적인 평가",
		1: "긍정적인 평가",
		2: "이도 저도 아닌 애매한 평가"
	}

	def __init__(
			self,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		# features - [Rating, Review, __index_level_0__ ]
		texts = dataset.data['Review']
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['Rating']]
		self._sentences: List[Sentence] = [
			Sentence(str(text), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for text, tag in zip(texts, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self._sentences[index]

	def __len__(self) -> int:
		return len(self._sentences)




class BaseMetaLearner(MetaLearner):

	def __init__(self, *args, **kwargs):
		super(BaseMetaLearner, self).__init__(*args, **kwargs)

	def base_learning(
			self,
			embedding: str = 'klue/bert-base',
			down_sample: float = 1.0,
			sample_missing_splits=False,
	):
		assert not self._tars_model and 0 < down_sample <= 1

		corpora = [
			fetch(KlueYnatDataset, sample_missing_splits=sample_missing_splits),
			fetch(KlueNliDataset, sample_missing_splits=sample_missing_splits),
			fetch(KlueStsDataset, sample_missing_splits=sample_missing_splits),
			fetch(PawsXDataset, sample_missing_splits=sample_missing_splits),
			fetch(NaverSentimentMovieCommentsDataset, sample_missing_splits=sample_missing_splits),
			fetch(KoreanRestaurantReviewsDataset, sample_missing_splits=sample_missing_splits),
		]

		if 0 < down_sample < 1.0:
			corpora = [c.downsample(percentage=down_sample) for c in corpora]

		# TODO: 추가 klue task, 한글 task
		# data = MultiCorpus(corpora, name='klue', sample_missing_splits=sample_missing_splits)

		tars = TARSClassifier(
			embeddings=embedding,
		)

		results = []
		for c in corpora:
			label_dict = c.make_label_dictionary(c.name)
			tars.add_and_switch_to_new_task(
				task_name=c.name,
				label_dictionary=label_dict,
				label_type=c.name,
				multi_label=label_dict.multi_label,
			)

			# initialize the text classifier trainer with corpus
			trainer = ModelTrainer(tars, c)

			# train model
			log_dir = self._output_path / 'tensorboard' / c.name
			log_dir.mkdir(parents=True, exist_ok=True)
			result = trainer.train(
				base_path=self._output_path / c.name,  # path to store the model artifacts
				learning_rate=self._learning_rate,  # use very small learning rate
				optimizer=AdamW,
				param_selection_mode=True,
				mini_batch_size=self._mini_batch_size,  # small mini-batch size since corpus is tiny
				patience=self._patience,
				max_epochs=self._max_epochs,  # terminate after 10 epochs
				train_with_dev=self._train_with_dev,
				use_tensorboard=True,
				tensorboard_log_dir=log_dir,
			)
			results.append(result)

		self._tars_model = tars  # replace with fine tuned model
		logger.info(f'fine tuning completed for corpora:{[c.name for c in corpora]}, results:{results}')
		return results


if __name__ == "__main__":
	meta = BaseMetaLearner(
		model_path=None,  # base learning
		max_epochs=30,
		mini_batch_size=32,
		train_with_dev=True
	)
	result = meta.base_learning(down_sample=0.3)
	path = meta.save_model()
	print(path)
