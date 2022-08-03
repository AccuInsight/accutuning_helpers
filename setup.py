from setuptools import setup, find_namespace_packages
setup(
    name="accutuning_helpers",
    version="1.0.32",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.19.5",
        "pandas==1.1.4",
        "scipy==1.5.2",
        "joblib==1.1.0",
        "scikit-learn==0.23.2",
        "xgboost==1.6.1",
        "lightgbm==3.3.2",
        "ctgan==0.4.3",
        "python-dateutil==2.8.2",
        # text added
        "konlpy==0.6",
        "customized_konlpy",
        "tweepy>=3.7.0",
        "langid>=1.1.6",
        "cleanlab>=1.0",
        "flair==0.11.3",
        "sentencepiece>=0.1.95",
        "datasets>=2.3.2", # Huggingface datasets
        "sentence-transformers>=2.2",
        # "protobuf==3.20.0",
        "conllu==4.4.1" # conllu== 4.4.2 로 update되면서 flair 11.3에서 문제 발생하여 hotfix 조치함. flair version up시 삭제할 것
    ]
)
