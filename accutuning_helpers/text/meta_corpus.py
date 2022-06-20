import logging
from typing import Dict, Literal, List

import datasets
from flair.data import Corpus, Sentence, Tokenizer
from flair.datasets.document_classification import FlairDataset
from flair.tokenization import SegtokTokenizer
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)
KlueTaskName = Literal['ynat', 'nli', 'ner', 're', 'dp', 'mrc', 'wos']


class KlueCorpus(Corpus):
	"""
	Corpus of Linguistic Acceptability from KLUE benchmark (https://klue-benchmark.com/tasks).
	KLUE is a collection of 8 tasks to evaluate natural language understanding capability of Korean language models.
	We delibrately select the 8 tasks, which are Topic Classification, Semantic Textual Similarity,
	Natural Language Inference, Named Entity Recognition, Relation Extraction, Dependency Parsing,
	Machine Reading Comprehension, and Dialogue State Tracking.
	"""

	def __init__(
			self,
			klue_task: KlueTaskName,
			tokenizer: Tokenizer = SegtokTokenizer(),
			**corpus_args
	):
		##TODO: introspection 이용한 huggingface dataset fetch generalization
		dataset_name = 'klue'
		corpus_ = datasets.load_dataset(dataset_name, klue_task)
		task_name = f"{dataset_name}-{klue_task}"
		if klue_task == 'ynat':
			train_ = KlueYnatDataset(task_name=task_name, dataset=corpus_['train'], tokenizer=tokenizer)
			dev_ = KlueYnatDataset(task_name=task_name, dataset=corpus_['validation'], tokenizer=tokenizer)
		elif klue_task == 'nli':
			train_ = KlueNliDataset(task_name=task_name, dataset=corpus_['train'], tokenizer=tokenizer)
			dev_ = KlueNliDataset(task_name=task_name, dataset=corpus_['validation'], tokenizer=tokenizer)
		else:
			raise NotImplementedError('other task is not yet implemented')

		super(KlueCorpus, self).__init__(
			name=task_name, train=train_, dev=dev_, **corpus_args
		)


class HuggingfaceDataset(FlairDataset):
	def is_in_memory(self) -> bool:
		return True


class KlueYnatDataset(HuggingfaceDataset):
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
			task_name: str,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		texts = dataset.data['title']  # features - [guid, title, label, url, date]
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self._sentences: List[Sentence] = [
			Sentence(str(text), use_tokenizer=tokenizer).add_label(task_name, tag)
			for text, tag in zip(texts, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self._sentences[index]

	def __add__(self, other: "KlueYnatDataset") -> ConcatDataset[Sentence]:
		return super().__add__(other)

	def __len__(self) -> int:
		return len(self._sentences)


class KlueNliDataset(HuggingfaceDataset):
	label_name_map: Dict[int, str] = {
		0: "관계 있음",
		1: "무관함",
		2: "모순",
	}

	def __init__(
			self,
			task_name: str,
			dataset: datasets.Dataset,
			tokenizer: Tokenizer,
	):
		##TODO: embedding tokenizer로부터 sep_token 읽어오는 로직 추가
		self.sep = " [SEP] "

		# features - [guid, source, premise, hypothesis, label]
		prems, hypos = dataset.data['premise'], dataset.data['hypothesis']
		tags = [self.label_name_map[i.as_py()] for i in dataset.data['label']]
		self.pairs: List[Sentence] = [
			Sentence(text=str(prem) + self.sep + str(hypo), use_tokenizer=tokenizer).add_label(task_name, tag)
			for prem, hypo, tag in zip(prems, hypos, tags)
		]

	def __getitem__(self, index: int) -> Sentence:
		return self.pairs[index]

	def __add__(self, other: "KlueNliDataset") -> ConcatDataset[Sentence]:
		return super().__add__(other)

	def __len__(self) -> int:
		return len(self.pairs)
