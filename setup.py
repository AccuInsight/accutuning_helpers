from setuptools import setup, find_namespace_packages
setup(
    name="accutuning_helpers",
    version="1.0.30",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.19.5",
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "scipy==1.5.2",
        "joblib==1.1.0",
        "python-dateutil==2.8.2",
        "ctgan==0.4.3",
        "xgboost==1.5.0",
        "lightgbm>=2.3.1"
        # text added
        "konlpy==0.6",
        "customized_konlpy",
        "tweepy>=3.7.0",
        "langid>=1.1.6",
        "cleanlab>=1.0",
        "flair==0.11.3",
        "sentencepiece>=0.1.95",
        # "datasets>2.3.2", # Huggingface datasets 
        "sentence-transformers==0.3.9",
        "protobuf==3.20.0",
    ]
)
