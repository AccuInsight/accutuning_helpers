from setuptools import setup, find_namespace_packages
setup(
    name="accutuning_helpers",
    version="1.0.27",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.18.1",
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "scipy==1.4.1",
        "joblib==0.16.0",
        "python-dateutil==2.8.1",
        "ctgan==0.3.1",
        "sentence-transformers==0.3.2",
    ]
)
