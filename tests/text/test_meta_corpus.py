from accutuning_helpers.text.meta_corpus import KlueStsDataset, fetch


def test_fetch_klue_sts():
	corpus = fetch(KlueStsDataset, sample_missing_splits=False)
	print(corpus)
	assert len(corpus.train) == 11668
