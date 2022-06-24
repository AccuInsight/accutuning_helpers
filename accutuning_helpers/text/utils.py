import logging
import pickle
from collections import Counter
from typing import List, Tuple

import cleanlab
import numpy as np
import pandas as pd
import torch
from flair.data import Label
from langid import langid
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from accutuning_helpers.text import NOT_CONFIDENT_TAG, MIN_LABEL_FREQUENCY
from accutuning_helpers.text.tokenizer import TwitterTokenizer

logger = logging.getLogger(__name__)


def tensor_to_array(vectors) -> np.ndarray:
	if torch.is_tensor(vectors):
		if vectors.requires_grad:
			vectors = vectors.detach()
		if vectors.is_cuda:
			vectors = vectors.cpu()
		vectors = vectors.numpy()
	return vectors


def load(filepath) -> pd.DataFrame:
	with open(filepath, 'rt') as f:
		if filepath.endswith('.csv'):
			df = pd.read_csv(filepath)
		elif filepath.endswith('.tsv'):
			df = pd.read_table(filepath)
		elif filepath.endswith('.xls') or filepath.endswith('.xlsx'):
			df = pd.read_excel(filepath)
		elif filepath.endswith('.txt'):
			texts = [line.strip() for line in f.readlines()]
			df = pd.DataFrame(texts, columns=['text'])
		elif filepath.endswith('.json'):
			df = pd.read_json(f, orient='table')
		elif filepath.endswith('.pkl'):
			df = pickle.load(f)
		else:
			raise Exception('unknown extension')

	return df


def identify_language(corpus_list: List[str], norm_probs=False) -> Tuple[List[str], List[float]]:
	identifier = langid.LanguageIdentifier.from_modelstring(langid.model, norm_probs=norm_probs)
	language = [identifier.classify(c)[0] for c in corpus_list]  # rank
	lang_id_result = dict(Counter(language))
	top_langs = list(lang_id_result)
	rst = list(lang_id_result.values())
	top_overall = [r / sum(rst) * 100 for r in rst]
	return top_langs, top_overall


def correct_label(texts: List[str], labels: List[Label]) -> Tuple[List[str], List[Label]]:
	top_langs, top_overall = identify_language(texts[:5])  # 최대 5문장만 보면 됨
	if top_langs[0] == 'ko' and top_overall[0] > 80:
		tokenized_stcs = [TwitterTokenizer().stcs_to_words(s) for s in texts]
	else:
		logger.debug(f"Identify Language - Top language:{top_langs[:1]}, Top overall:{top_overall[:1]}")
		tokenized_stcs = texts

	tfidf = TfidfVectorizer().fit(tokenized_stcs)
	embs = tfidf.transform(tokenized_stcs).toarray()

	tags = [x.value for x in labels]
	tags = sorted(list(set(tags)))
	label2int = {l: i for i, l in enumerate(tags)}
	int2label = {i: l for i, l in enumerate(tags)}
	int_labels = [label2int[t] for t in tags]
	int_y = int_labels

	X = np.array(embs)
	s = np.array(int_y)

	psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
		X, s, clf=LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs'))
	pyx = psx
	logger.debug("Fetched probabilities for", pyx.shape[0], 'examples and', pyx.shape[1], 'classes.')

	# Estimate the confident joint, a proxy for the joint distribution of label noise.
	cj, cj_only_label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
		s, pyx,
		return_indices_of_off_diagonals=True,
	)

	label_errors_idx = cleanlab.pruning.get_noise_indices(
		s=s,
		psx=pyx,
		confident_joint=cj,
		prune_method='both',
		sorted_index_method='normalized_margin',  # ['prob_given_label', 'normalized_margin']
	)
	replacements = []
	for idx in label_errors_idx:
		replacements.append(int2label[np.argmax(pyx[idx])])
	for (index, replacement) in zip(label_errors_idx, replacements):
		labels[index].value = replacement
	return texts, labels


def sampling(df: pd.DataFrame, tag_column_name: str, class_nm_list: List[str], n_samples: int) -> pd.DataFrame:
	replace = True  # with replacement
	fn = lambda obj: obj.loc[np.random.choice(obj.index, n_samples, replace), :]
	rst = df.loc[df[tag_column_name].isin(class_nm_list)]
	rst = rst.groupby(tag_column_name, as_index=False).apply(fn)
	return rst


def evaluate(gold: List[str], pred: List[str]) -> None:
	counter = Counter(pred)
	print(f'분류 못함: {counter[NOT_CONFIDENT_TAG]} / {len(gold)} 건 포함 metrics')
	print(metrics.classification_report(gold, pred))

	pred_expt = []
	tags_expt = []
	for idx, p in enumerate(pred):
		if p == NOT_CONFIDENT_TAG:
			pass
		else:
			pred_expt.append(pred[idx])
			tags_expt.append(gold[idx])
	print(f'분류 못함: {counter[NOT_CONFIDENT_TAG]} / {len(gold)} 건 제외 metrics')
	print(metrics.classification_report(tags_expt, pred_expt))


def is_confident_in_label_prediction(tags: List[str], class_name_list: List) -> bool:
	total_cnt = len(tags)  # 전체 레이블 문장 건수
	dict_cls_num = Counter(tags)  # 레이블 결과 건수

	prop = dict_cls_num[NOT_CONFIDENT_TAG] / total_cnt  # 전체 건수 대비 '분류못함' 레이블의 건수 비율:  0.7 이상이면 제로샷 결과만 리턴
	if prop > 0.7:  # 분류못함이 70% 이상
		return False

	if any(filter(lambda x: (x not in dict_cls_num) or (dict_cls_num[x] <= MIN_LABEL_FREQUENCY), class_name_list)):
		return False
	return True
