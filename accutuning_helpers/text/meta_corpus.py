import logging
import math
from copy import copy
from datetime import datetime
from typing import Dict, List

import datasets
from flair.data import Corpus, Sentence, Tokenizer
from flair.datasets.document_classification import FlairDataset
from flair.models import TARSClassifier
from flair.optim import LinearSchedulerWithWarmup
from flair.tokenization import SegtokTokenizer
from flair.trainers import ModelTrainer
from torch.optim import AdamW

from accutuning_helpers.text.meta_learning import MetaLearner

logger = logging.getLogger(__name__)


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
		_train = cls(dataset=train_, tokenizer=tokenizer)
	if dev_:
		_dev = cls(dataset=dev_, tokenizer=tokenizer)
	if test_:
		_test = cls(dataset=test_, tokenizer=tokenizer)

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
	dataset_args = ['paws-x', 'ko']
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
			Sentence(str(text).strip(), use_tokenizer=tokenizer).add_label(self.task_name, tag)
			for text, tag in zip(texts, tags) if str(text).strip()
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

	def __init__(self, *args, base_language='ko', **kwargs):
		super(BaseMetaLearner, self).__init__(*args, **kwargs)
		self._lang = base_language

	def base_learning(
			self,
			embedding: str = 'klue/bert-base',
			down_sample: float = 1.0,
			sample_missing_splits=False,
			corpus_iteration: int = 3,
	):
		assert not self._tars_model and 0 < down_sample <= 1

		corpora = [
			fetch(KlueYnatDataset, sample_missing_splits=sample_missing_splits),
			fetch(KlueNliDataset, sample_missing_splits=sample_missing_splits),
			fetch(KlueStsDataset, sample_missing_splits=sample_missing_splits),
			fetch(PawsXDataset, sample_missing_splits=sample_missing_splits),
			fetch(NaverSentimentMovieCommentsDataset, sample_missing_splits=sample_missing_splits),
			fetch(KoreanRestaurantReviewsDataset, sample_missing_splits=sample_missing_splits),
			# NEWSGROUPS(sample_missing_splits=sample_missing_splits)
		]

		tars = TARSClassifier(
			embeddings=embedding,
		)
		# optimizer_params
		_params = list(tars.tars_model.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		decay = 0.01
		params = [
			{'params': [p for n, p in _params if not any(nd in n for nd in no_decay)], 'weight_decay': decay},
			{'params': [p for n, p in _params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(params, lr=self._learning_rate, weight_decay=decay)
		# optimizer = Adam
		results = []
		mmdd = datetime.now().strftime("%m%d_%H%M")
		for i in range(1, corpus_iteration + 1):
			for c in corpora:
				if 0 < down_sample < 1.0:
					c = copy(c).downsample(percentage=down_sample)

				logger.info(f" start training for corpus {c.name}, {i} -- iteration")
				# tensorboard log directory
				log_dir = self._output_path / 'tensorboard' / mmdd / f'{c.name}_{i}'
				log_dir.mkdir(parents=True, exist_ok=True)

				if c.name in tars.list_existing_tasks():
					tars.switch_to_task(c.name)
				else:
					label_dict = c.make_label_dictionary(c.name)
					tars.add_and_switch_to_new_task(
						task_name=c.name,
						label_dictionary=label_dict,
						label_type=c.name,
						multi_label=label_dict.multi_label,
					)

				# initialize the text classifier trainer with corpus
				total_steps = math.ceil(len(c.train) / self._mini_batch_size) * self._max_epochs
				scheduler = LinearSchedulerWithWarmup(
					optimizer=optimizer,
					num_train_steps=total_steps,
					num_warmup_steps=self._warmup_fraction * total_steps,
				)

				trainer = ModelTrainer(tars, c)
				result = trainer.train(
					base_path=self._output_path / mmdd / c.name,  # path to store the model artifacts
					learning_rate=self._learning_rate,  # use very small learning rate
					# optimizer=AdamW,
					# optimizer=Adam, # default SGD
					optimizer=optimizer,
					scheduler=scheduler,
					mini_batch_size=self._mini_batch_size,  # small mini-batch size since corpus is tiny
					patience=self._patience,
					warmup_fraction=self._warmup_fraction,
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
		max_epochs=20,
		mini_batch_size=16,
		mini_batch_chunk_size=4,
		# learning_rate=1e-4,
		learning_rate=7e-5,
		# learning_rate=5e-5,  # learning rate
		# learning_rate=5e-3,
		# learning_rate=0.02,
		train_with_dev=False,
		base_language='ko'
	)
	# result = meta.base_learning(down_sample=1.0, embedding="kykim/bert-kor-base")
	result = meta.base_learning(down_sample=0.1, embedding="kykim/electra-kor-base")
	# result = meta.base_learning(down_sample=0.5, embedding="klue/bert-base")
	# result = meta.base_learning(down_sample=0.1, embedding="bert-base-cased")
	path = meta.save_model()
	print(path)
