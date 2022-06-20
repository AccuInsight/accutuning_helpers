import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Union, Tuple

import pandas as pd
import transformers.optimization
from flair.data import Sentence, Corpus, Label
from flair.datasets import FlairDatapointDataset
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

from accutuning_helpers.text import NOT_CONFIDENT_TAG
from accutuning_helpers.text import utils as labeler_utils
from accutuning_helpers.text.meta_corpus import KlueCorpus

logger = logging.getLogger(__name__)


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


META_TRAINED_MODEL_PATH = '/code/resources/taggers/agnews_all/final-model.pt'
DEFAULT_TAG_COLUMN_NAME = 'predicted_tags'
WORKPLACE_HOME = os.environ.get('ACCUTUNING_WORKSPACE_ROOT', '/workspace')
WORKPLACE_PATH = os.environ['ACCUTUNING_WORKSPACE']
PREDICTION_SCORE_THREASHOLD = 0.7


def to_predictions(sentences: List[Sentence]) -> List[Tuple[Label, float]]:
	predictions = []
	for sentence in sentences:
		number_list = [label.score for label in sentence.labels]
		max_value = max(number_list) if number_list else 0
		if len(number_list) > 0:
			# if max_value >= PREDICTION_SCORE_THREASHOLD:
			max_index = number_list.index(max_value)
			max_label, max_score = sentence.labels[max_index], sentence.labels[max_index].score
			predictions.append((max_label, max_score,))
		else:
			not_confident = Label(sentence, value=NOT_CONFIDENT_TAG, score=0.5)
			predictions.append((not_confident, not_confident.score,))  ## TODO: NOT Confident 분기 정교화?
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
			model_path: Union[str, Path] = META_TRAINED_MODEL_PATH,
			lang: str = 'ko',
			learning_rate=5e-5,
			mini_batch_size=1,
			patience=10,
			max_epochs=1,
			train_with_dev=True,
			n_samples=5,
			prediction_batch_size=4,
	):
		self._lang = lang
		self._learning_rate = learning_rate
		self._mini_batch_size = mini_batch_size
		self._patience = patience
		self._max_epochs = max_epochs
		self._train_with_dev = train_with_dev
		self._n_samples = n_samples
		self._batch_size = prediction_batch_size

		self._output_path = output_path or Path(WORKPLACE_PATH, 'output')

		##TODO: detect language -> pretrained model 선택, fine tuned model 이 있는지 여부 탐색?
		self._model_path = model_path
		if self._model_path:
			self._tars_model: TARSClassifier = TARSClassifier.load(self._model_path)
		else:
			self._tars_model: TARSClassifier = None

	def base_learning(
			self,
			embedding: str = 'klue/bert-base',
			down_sample: float = 1.0,
			sample_missing_splits=False,
	):
		assert not self._tars_model and 0 < down_sample <= 1

		if 0 < down_sample < 1.0:
			corpora = [
				# KlueCorpus(klue_task='ynat', sample_missing_splits=sample_missing_splits).downsample(down_sample),
				KlueCorpus(klue_task='nli', sample_missing_splits=sample_missing_splits).downsample(down_sample),
			]
		else:  # down_sample == 1.0
			corpora = [
				KlueCorpus(klue_task='ynat', sample_missing_splits=sample_missing_splits),
				KlueCorpus(klue_task='nli', sample_missing_splits=sample_missing_splits),
			]

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
			result = trainer.train(
				base_path=self._output_path / c.name,  # path to store the model artifacts
				learning_rate=self._learning_rate,  # use very small learning rate
				mini_batch_size=self._mini_batch_size,  # small mini-batch size since corpus is tiny
				patience=self._patience,
				max_epochs=self._max_epochs,  # terminate after 10 epochs
				train_with_dev=self._train_with_dev,
				use_tensorboard=True,
				tensorboard_log_dir=self._output_path / 'tensorboard'/ c.name,
			)
			results.append(result)

		self._tars_model = tars  # replace with fine tuned model
		logger.info(f'fine tuning completed for corpora:{[c.name for c in corpora]}, results:{results}')
		return results

	@timer
	def fine_tuning(
			self,
			df: pd.DataFrame,
			text_column_name: str = 'stcs',
			tag_column_name: str = 'tags',
			task_name: str = None,
	) -> Dict[str, str]:
		tars = self._tars_model

		texts = df[text_column_name].values.tolist()
		tags = df[tag_column_name].values.tolist()
		task_name = task_name or get_task_name(tags)

		tr = [Sentence(text).add_label(task_name, tag) for text, tag in zip(texts, tags)]
		dataset = FlairDatapointDataset(tr)
		corpus = Corpus(train=dataset, sample_missing_splits=False)

		if task_name in tars.list_existing_tasks():
			tars.switch_to_task(task_name)
		else:
			label_dict = c.make_label_dictionary(task_name)
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
			mini_batch_size=self._mini_batch_size,  # small mini-batch size since corpus is tiny
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
	) -> List[Tuple[str, float]]:
		tars = self._tars_model
		batch = self._batch_size

		sentences = [Sentence(text) for text in texts]
		if class_nm_list:  # zero shot
			for i in range(0, len(sentences), batch):
				tars.predict_zero_shot(sentences[i: i + batch], class_nm_list)
		else:
			tars.predict(sentences, mini_batch_size=batch)

		return to_predictions(sentences)

	@timer
	def zero_shot_learning(
			self,
			class_nm_list: List[str],
			target_column_nm: str,
			source_data_fp: str,
			**config_kwargs,
	) -> Dict[str, Union[str, List[str]]]:
		input_path = os.path.join(WORKPLACE_PATH, source_data_fp)
		target_df = labeler_utils.load(input_path)
		texts = target_df[target_column_nm].values.tolist()

		predictions = self._shot_learning(texts, class_nm_list)

		# rst_df = pd.read_csv(output_path / 'zsl_rst_df.csv')

		# TODO: 공구리 치기
		# if accutuning_lb.is_confident_in_label_prediction(predictions, class_nm_list):
		# 	# sample 결과를 가지고 FSL하여 Fine-tuning 후 저장
		# 	sample_df = labeler_utils.sampling(rst_df, DEFAULT_TAG_COLUMN_NAME, class_nm_list,
		# 									   n_samples=self._n_samples)
		# 	ret = fsl(rst_df, 'stcs', sample_df, 'stcs', 'tags', output_path, conf_path)
		# 	rst_df = pd.read_csv(output_path / 'fsl_rst_df.csv')
		# 	ret['fine_tuned_model'] = fine_tuning(output_path, rst_df)

		return {
			'text_name': target_column_nm,
			'texts': texts,
			'tag_name': target_column_nm,
			'predictions': list(map(lambda x: x[0], predictions)),  # value만, score 제외
		}

	@timer
	def few_shot_learning(
			self,
			target_column_nm,
			source_data_fp,
			samples_target_column_nm,
			samples_tag_column_nm,
			samples_fp,
			correct,
			**config_kwargs,
	) -> Dict[str, Union[str, List[str]]]:

		# 1. fine tuning with samples
		sample_path = os.path.join(WORKPLACE_HOME, samples_fp)
		samples_df = labeler_utils.load(sample_path)
		task_name = config_kwargs.get('task_name')
		self.fine_tuning(samples_df, samples_target_column_nm, samples_tag_column_nm, task_name=task_name)

		# 2. predict whole data
		input_path = os.path.join(WORKPLACE_HOME, source_data_fp)
		target_df = labeler_utils.load(input_path)
		texts = target_df[target_column_nm].values.tolist()
		predictions = self._shot_learning(texts, class_nm_list=None)

		# 3. correct predictions if told to do so
		if correct:
			logger.debug('FSL - Correcting labels')
			texts, predictions = labeler_utils.correct_label(texts, predictions)

		# 4. return result
		return {
			'text_name': target_column_nm,
			'texts': texts,
			'tag_name': DEFAULT_TAG_COLUMN_NAME,
			'predictions': list(map(lambda x: x[0], predictions)),  # value만, score 제외
		}

	@timer
	def label_predict(
			self,
			bulk_output_fp,
			fine_tuned_model_fp,
			class_nm_list,
			target_column_nm,
			source_data_fp,
			**config_kwargs,
	):
		result: Dict = self.zero_shot_learning(
			class_nm_list,
			target_column_nm,
			source_data_fp,
			**config_kwargs,
		)
		return result

	# command = 'python /code/accutuning_lb/zsl_predict.py --conf_path %s --model_path %s --output_fp %s' % (
	# 	conf_path, fine_tuned_model_fp, bulk_output_fp)
	# process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	# process.wait()
	#
	# end_t = datetime.datetime.now()
	#
	# elapsed_time = end_t - start_t
	# logger.info(f'Labeling predict - Finished, Elapsed time {elapsed_time}')
	#
	# return {
	# 	"total_duration": elapsed_time.microseconds,
	# 	"pred_duration": (end_t - mid_t).microseconds,
	# 	"bulk_output_fp": str(bulk_output_fp),
	# }

	def save_model(self, file_path=None) -> str:
		file_path = file_path or str(self._output_path / f'fine_tuned_{self._lang}.pt')
		self._tars_model.save(model_file=file_path)
		logger.info(f'model saved in the path:{file_path}')
		return file_path

	def save_result(
			self,
			result_file: str,
			text_name: str,
			texts: List[str],
			tag_name: str,
			predictions: List[str],
	) -> Dict[str, str]:
		output_path = self._output_path
		# save results
		result_df = pd.DataFrame({text_name: texts, tag_name: predictions})
		result_df.to_csv(output_path / result_file, index=False)
		labeler_utils.save_output_file(output_path / 'labels.pkl', predictions)
		labeler_utils.save_output_file(output_path / 'clusters.pkl', list(set(predictions)))

		# TODO: save model location
		output_location = {
			'labels': str((output_path / 'labels.pkl').relative_to(WORKPLACE_PATH)),
			'clusters': str((output_path / 'clusters.pkl').relative_to(WORKPLACE_PATH)),
			'fine_tuned_model': self.save_model()
		}

		# save output location
		(output_path / 'output.json').write_text(
			json.dumps(output_location)
		)
		return output_location


if __name__ == "__main__":
	# meta = MetaLearner(
	# 	model_path=None,  # base learning
	# 	max_epochs=3,
	# 	mini_batch_size=1,
	# 	train_with_dev=True
	# )
	# result = meta.base_learning(down_sample=0.001, sample_missing_splits=True)

	meta = MetaLearner(
		model_path=None,  # base learning
		max_epochs=20,
		mini_batch_size=32,
		train_with_dev=True
	)
	result = meta.base_learning(down_sample=0.1, sample_missing_splits=True)
	path = meta.save_model()
	print(path)
