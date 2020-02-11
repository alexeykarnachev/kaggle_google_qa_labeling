from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def get_version(filename):
    from re import findall
    with open(filename) as f:
        metadata = dict(findall("__([a-z]+)__ = '([^']+)'", f.read()))
    return metadata['version']


setup(
    name='kaggle_google_qa_labeling',
    version=get_version('kaggle_google_qa_labeling/__init__.py'),
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        "torch==1.3.1",
        "transformers==2.4.1",
        "scikit-learn==0.21.3",
        "pandas==0.25.1",
        "future==0.17.1",
        "tqdm==4.36.1"
    ]
)
