from setuptools import find_namespace_packages, setup

setup(
    name="accutuning_helpers",
    version="1.1.0",
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
        "python-dateutil==2.8.2"
    ]
)
