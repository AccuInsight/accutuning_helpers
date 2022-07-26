import json
import logging
import os
import pickle
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Union

import pandas as pd
from flair.data import Sentence, Corpus, Label
from flair.datasets import FlairDatapointDataset
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from torch.optim.adamw import AdamW

from accutuning_helpers.text import NOT_CONFIDENT_TAG
from accutuning_helpers.text import utils as labeler_utils

logger = logging.getLogger(__name__)

META_MODEL_DIR = '/code/resources/labeler_models/'
META_MODEL_BIN_KO = 'tars-bert-ko-v1.pt'
META_MODEL_BIN_EN = 'tars-base-v8.pt'

DEFAULT_TAG_COLUMN_NAME = 'gold_tags'
WORKPLACE_HOME = os.environ.get('ACCUTUNING_WORKSPACE_ROOT', '/workspace')
WORKPLACE_PATH = os.environ['ACCUTUNING_WORKSPACE']
PREDICTION_SCORE_THRESHOLD = 0.7


def timer(fn):
	"""
	timer decorator
	"""

	def inner(*args, **kwargs):
		logger.info(f'# Start {fn.__name__}')
		start_time = perf_counter()
		to_execute = fn(*args, **kwargs)
		end_time = perf_counter()
		execution_time = end_time - start_time
		logger.info(f'## {fn.__name__} took {execution_time * 100:.3f}ms to execute')
		result: Dict = to_execute
		result['start_time'] = start_time
		result['end_time'] = end_time
		result['execution_time'] = execution_time
		return result

	return inner


def save_output_file(filepath: Path, obj) -> Union[str, Path]:
	filepath.write_bytes(
		pickle.dumps(obj)
	)
	return filepath


def _relative_to_workspace(filepath: Union[str, Path]) -> str:
	if filepath is None:
		return filepath

	if isinstance(filepath, str):
		filepath = Path(filepath)
	if str(filepath).startswith(META_MODEL_DIR):
		return str(filepath.relative_to(META_MODEL_DIR))
	return str(filepath.relative_to(WORKPLACE_HOME))


def to_predictions(sentences: List[Sentence]) -> List[Label]:
	predictions = []
	for sentence in sentences:
		number_list = [label.score for label in sentence.labels]
		max_value = max(number_list) if number_list else 0
		if len(number_list) > 0:
			# if max_value >= PREDICTION_SCORE_THREASHOLD:
			max_index = number_list.index(max_value)
			max_label = sentence.labels[max_index]
			predictions.append(max_label)
		else:
			not_confident = Label(sentence, value=NOT_CONFIDENT_TAG, score=0.5)
			predictions.append(not_confident)  ## TODO: NOT Confident 분기 정교화?
	return predictions


def get_task_name(labels: List[str]) -> str:
	"""
	##FIXME: tricky - 명시적으로 task name을 받는게 좋음
	특정 task name이 없는 경우, unique label 의 집합을 task name 으로 한다
	"""
	unique = sorted(set(labels))
	return '__'.join(unique)


class MetaLearner:

	def __init__(
			self,
			output_path: Path = None,
			model_path: Union[str, Path] = None,
			tars_model: TARSClassifier = None,
			learning_rate=5e-5,
			mini_batch_size=16,
			mini_batch_chunk_size=8,
			warmup_fraction=0.1,
			patience=10,
			max_epochs=10,
			train_with_dev=True,
			n_samples=5,
			prediction_batch_size=16,
	):
		self._learning_rate = learning_rate
		self._mini_batch_size = mini_batch_size
		self._mini_batch_chunk_size = mini_batch_chunk_size
		self._warmup_fraction = warmup_fraction
		self._patience = patience
		self._max_epochs = max_epochs
		self._train_with_dev = train_with_dev
		self._n_samples = n_samples
		self._prediction_batch_size = prediction_batch_size
		self._output_path = output_path or Path(WORKPLACE_PATH, 'output')
		self._model_path = model_path
		self._lang = None  # lazy identify
		self._tars_model: TARSClassifier = tars_model  # lazy loading

	@property
	def model_path(self):
		"""
		현재 self._tars_model 이 load 되었던 directory를 return 한다.
		load 이후 학습이 진행된 경우 weight가 달라졌으므로 이에 유의한다.
		"""
		return self._model_path

	def _load_model(self, model_path: str = None, lang: str = 'ko'):
		"""
		언어가 결정되는 시점은 data를 읽은 이후라, 늦게 모델을 load 한다
		"""
		model = self._tars_model
		if not model:
			if not model_path:
				if lang == 'ko':
					model_path = os.path.join(META_MODEL_DIR, META_MODEL_BIN_KO)
				else:  # english 취급
					model_path = os.path.join(META_MODEL_DIR, META_MODEL_BIN_EN)

			model = TARSClassifier.load(model_path)
			self._tars_model = model
			self._model_path = model_path
			self._lang = lang
		return model

	def _identify_language(self, texts: List[str]):
		lang = self._lang
		if not lang:
			langs, _ = labeler_utils.identify_language(texts)
			lang = langs[0] if langs else 'en'  # default
			self._lang = lang
		return lang

	@timer
	def fine_tuning(
			self,
			df: pd.DataFrame = None,
			text_column_name: str = None,
			tag_column_name: str = None,
			texts: List[str] = None,
			tags: List[str] = None,
			task_name: str = None,
	) -> Dict[str, str]:
		assert df is not None or texts is not None, "df or text should be provided."

		texts = texts or df[text_column_name].values.tolist()
		tags = tags or df[tag_column_name].values.tolist()
		task_name = task_name or get_task_name(tags)

		tr = [Sentence(text).add_label(task_name, tag) for text, tag in zip(texts, tags)]
		dataset = FlairDatapointDataset(tr)
		corpus = Corpus(train=dataset, sample_missing_splits=False)

		lang = self._identify_language(texts)
		tars = self._load_model(model_path=self.model_path, lang=lang)

		if task_name in tars.list_existing_tasks():
			tars.switch_to_task(task_name)
		else:
			label_dict = corpus.make_label_dictionary(task_name)
			tars.add_and_switch_to_new_task(
				task_name=task_name,
				label_dictionary=label_dict,
				label_type=task_name,
				multi_label=label_dict.multi_label,
			)

		# initialize the text classifier trainer with corpus
		trainer = ModelTrainer(tars, corpus)

		# train model
		result = trainer.fine_tune(
			base_path=self._output_path,  # path to store the model artifacts
			learning_rate=self._learning_rate,  # use very small learning rate
			optimizer=AdamW,
			param_selection_mode=True,
			mini_batch_size=self._mini_batch_size,  # small mini-batch size since corpus is tiny
			mini_batch_chunk_size=self._mini_batch_chunk_size,
			max_epochs=self._max_epochs,  # terminate after 10 epochs
			train_with_dev=self._train_with_dev
		)
		self._tars_model = tars  # replace with fine tuned model
		logger.info(f'fine tuning result:{result}')
		return result

	def _shot_learning(
			self,
			texts: List[str],
			class_nm_list: List[str] = None,
	) -> List[Label]:
		lang = self._identify_language(texts)
		tars = self._load_model(model_path=self.model_path, lang=lang)

		sentences = [Sentence(text) for text in texts]
		if class_nm_list:  # zero shot
			tars.predict_zero_shot(sentences, class_nm_list)
		else:
			# already tasks are added at the fine tuning time
			tars.predict(sentences, mini_batch_size=self._prediction_batch_size)

		return to_predictions(sentences)

	@timer
	def zero_shot_learning(
			self,
			class_nm_list: List[str],
			target_column_nm: str = None,
			source_data_fp: str = None,
			texts: List[str] = None,
			tag_column_nm: str = None,
			**config_kwargs,
	) -> Dict[str, Union[str, List[str], List[Label]]]:
		assert texts or source_data_fp, 'texts or source data file should be provided.'

		if not texts:
			input_path = os.path.join(WORKPLACE_HOME, source_data_fp)
			target_df = labeler_utils.load(input_path)
			texts = target_df[target_column_nm].values.tolist()
		predictions = self._shot_learning(texts, class_nm_list)

		return {
			'text_name': target_column_nm,
			'texts': texts,
			'tag_name': tag_column_nm,
			'predictions': predictions,  # value만, score 제외
			'class_nm_list': class_nm_list,
		}

	@timer
	def few_shot_learning(
			self,
			target_column_nm: str = None,
			tag_column_nm: str = None,
			source_data_fp: str = None,
			texts: List[str] = None,
			samples_target_column_nm: str = None,
			samples_tag_column_nm: str = None,
			samples_fp: str = None,
			sample_texts: List[str] = None,
			sample_tags: List[str] = None,
			task_name: str = None,
			correct: bool = False,
			**config_kwargs,
	) -> Dict[str, Union[str, List[str], List[Label]]]:
		assert sample_texts or samples_fp, "sample texts or sample file should be provided."

		# 1. fine tuning with samples
		if sample_texts and sample_tags:
			self.fine_tuning(texts=sample_texts, tags=sample_tags, task_name=task_name)
		else:
			sample_path = os.path.join(WORKPLACE_HOME, samples_fp)
			samples_df = labeler_utils.load(sample_path)
			self.fine_tuning(samples_df, samples_target_column_nm, samples_tag_column_nm, task_name=task_name)

		# 2. predict whole data
		assert texts or source_data_fp, 'texts or source data file should be provided.'
		if not texts:
			input_path = os.path.join(WORKPLACE_HOME, source_data_fp)
			target_df = labeler_utils.load(input_path)
			texts = target_df[target_column_nm].values.tolist()
		predictions = self._shot_learning(texts, class_nm_list=None)

		# 3. correct predictions if told to do so
		# FIXME: cleanlab 2.0 적용. 분류못함 삭제되도록 유도 적용
		# if correct:
		# 	logger.debug('FSL - Correcting labels')
		# 	texts, predictions = labeler_utils.correct_label(texts, predictions)

		# 4. return result
		return {
			'text_name': target_column_nm,
			'texts': texts,
			'tag_name': tag_column_nm,
			'predictions': predictions,  # value만, score 제외
		}

	@timer
	def label_predict(
			self,
			class_nm_list,
			target_column_nm,
			source_data_fp,
			**config_kwargs,
	) -> Dict[str, Union[str, List[str], List[Label]]]:
		result: Dict = self.zero_shot_learning(
			class_nm_list,
			target_column_nm,
			source_data_fp,
			**config_kwargs,
		)
		return result

	def save_model(self, file_path=None) -> str:
		file_path = file_path or str(self._output_path / f'fine_tuned_{self._lang}.pt')
		self._tars_model.save(model_file=file_path)
		logger.info(f'model saved in the path:{file_path}')
		return file_path

	def save_result(
			self,
			result_csv_filename: str,
			text_name: str,
			texts: List[str],
			tag_name: str,
			predictions: List[Label],
			execution_time: float,
			tags: List[Union[int, str]] = None,
			model_path: str = None,
			**config_kwargs,  # misc arguments - ignored
	) -> Dict[str, str]:
		output_path = self._output_path
		tag_name = tag_name or DEFAULT_TAG_COLUMN_NAME

		# save results
		result = {}
		result[text_name] = texts
		if tags:  # gold labels
			result[tag_name] = tags

		p_labels = [label.value for label in predictions]
		p_scores = [label.score for label in predictions]
		result[f'{tag_name}_predicted'] = p_labels
		result['confidence'] = p_scores

		result_df = pd.DataFrame(result)
		result_df.to_csv(output_path / result_csv_filename, index=False)

		labels_path = save_output_file(output_path / 'labels.pkl', (p_labels, p_scores))
		clusters_path = save_output_file(output_path / 'clusters.pkl', list(set(p_labels)))

		output_path_info = {
			'output_fp': _relative_to_workspace(output_path / result_csv_filename),
			'labels': _relative_to_workspace(labels_path),
			'clusters': _relative_to_workspace(clusters_path),
			'fine_tuned_model': _relative_to_workspace(model_path),
			'execution_time': execution_time,
		}

		# save output location
		(output_path / 'output.json').write_text(
			json.dumps(output_path_info)
		)
		return output_path_info
