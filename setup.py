from setuptools import setup

setup(
    name='deepdive',
    version='0.1.0',
    description='package for deep net feature extraction and benchmarking',
    author = 'Colin Conwell',
    author_email='colinconwell@gmail.com',
    install_requires = ['numpy','pandas','matplotlib','tdqm', 'torch', 'torchvision', 'sklearn', 'timm', 'visualpriors'],
    url = 'https://github.com/ColinConwell/DeepDive',
)
