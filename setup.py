from setuptools import setup, find_namespace_packages
setup(
    name="accutuning_helpers",
    version="1.0.29",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.21.4",
        "pandas==1.3.4",
        "scikit-learn==1.0.1",
        "scipy==1.7.2",
        "joblib==1.1.0",
        "python-dateutil==2.8.2",
        # "ctgan==0.5.0",
        "sentence-transformers==0.3.9",
    ]
)
