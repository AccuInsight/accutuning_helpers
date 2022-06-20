from unittest import TestCase

from accutuning_helpers.text.meta_learning import MetaLearner


class TestMetaLearner(TestCase):

	def setUp(self) -> None:
		self.meta = MetaLearner(
			model_path='tars-base',
			max_epochs=10,
			train_with_dev=True,
		)

	def test_fine_tuning(self):
		assert False

	def test_zero_shot_learning(self):
		conf = {
			"random_seed": 42,
			"source_data_fp": "data/nnst_lt_10.csv",
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
		print(result)
		assert all(filter(lambda x: x.value in conf['class_nm_list'], result['predictions']))

	def test_few_shot_learning(self):
		conf = {
			"random_seed": 42,
			"source_data_fp": "data/nnst_lt_10.csv",
			"samples_fp": "data/nnst_lt_10.csv",
			"related_stcs": "",
			"correct": True,
			"class_nm_list": None,
			"target_column_nm": "stcs",
			"samples_target_column_nm": "stcs",
			"samples_tag_column_nm": "tags",
			"labeler_worker_type": "fsl",
		}
		meta = self.meta
		result = meta.few_shot_learning(**conf)
		print(result)
		assert result

	def test_label_predict(self):
		assert False


class TestMetaBaseLearner(TestCase):

	def setUp(self) -> None:
		self.meta = MetaLearner(
			model_path=None,
			max_epochs=10,
			train_with_dev=True,
		)

	def test_base_learning(self):
		result = self.meta.base_learning(down_sample=0.1)
		assert result
		path = self.meta.save_model()
