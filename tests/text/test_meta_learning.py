import os
import json
from unittest import TestCase

from accutuning_helpers.text import utils as labeler_utils
from accutuning_helpers.text.meta_learning import MetaLearner, WORKPLACE_PATH


class TestMetaLearner(TestCase):

	def setUp(self) -> None:
		self.meta = MetaLearner(
			model_path=os.path.join(WORKPLACE_PATH, 'output/fine_tuned_ko.pt'),
			max_epochs=10,
			train_with_dev=True,
		)

	# def test_fine_tuning(self):
	# 	assert False
	#
	def test_zero_shot_learning(self):
		conf = {
			"random_seed": 42,
			"source_data_fp": "data/nnst_lt_10.csv",
			# "source_data_fp": "data/nnst_lt_1990.csv",
			"samples_fp": "",
			"related_stcs": "",
			"correct": True,
			"class_nm_list": ["생활", "기술"],
			"target_column_nm": "stcs",
			"samples_target_column_nm": "내용",
			"samples_tag_column_nm": "라벨",
			"labeler_worker_type": "zsl"
		}
		meta = self.meta
		result = meta.zero_shot_learning(**conf)
		result.pop('texts') # too long to print
		result['predictions'] = [p.value for p in result['predictions']]
		print(json.dumps(result, indent=2, ensure_ascii=False),)
		# assert all(map(lambda x: x in conf['class_nm_list'], result['predictions']))

		input_path = os.path.join(WORKPLACE_PATH, conf['source_data_fp'])
		target_df = labeler_utils.load(input_path)
		gold, pred = target_df['tags'], result['predictions']
		labeler_utils.evaluate(gold, pred)

	def test_few_shot_learning(self):
		conf = {
			"random_seed": 42,
			"source_data_fp": "data/nnst_lt_10.csv",
			"samples_fp": "data/nnst_lt_10.csv",
			"related_stcs": "",
			"correct": False,
			"class_nm_list": None,
			"target_column_nm": "stcs",
			"samples_target_column_nm": "stcs",
			"samples_tag_column_nm": "tags",
			"labeler_worker_type": "fsl",
		}
		meta = self.meta
		result = meta.few_shot_learning(**conf)
		result.pop('texts') # too long to print
		result['predictions'] = [p.value for p in result['predictions']]
		print(json.dumps(result, indent=2, ensure_ascii=False),)

		input_path = os.path.join(WORKPLACE_PATH, conf['source_data_fp'])
		target_df = labeler_utils.load(input_path)
		gold, pred = target_df['tags'], result['predictions']
		labeler_utils.evaluate(gold, pred)

		assert result

	# def test_label_predict(self):
	# 	assert False


class TestMetaBaseLearner(TestCase):

	def setUp(self) -> None:
		self.meta = MetaLearner(
			model_path=None,
			max_epochs=10,
			train_with_dev=True,
		)

	def test_base_learning(self):
		result = self.meta.base_learning(down_sample=0.01)
		assert result
		path = self.meta.save_model()
		assert os.path.exists(path)
