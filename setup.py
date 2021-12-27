import setuptools as setuptools

setuptools.setup(
    name='DeepDive',
    version='0.1.0',
    packages=['DeepDive'],
    package_data={'': ['model_opts/model_metadata.csv','model_opts/model_typology.csv']},
    description='package for deep net feature extraction and benchmarking',
    author = 'Colin Conwell',
    author_email='colinconwell@gmail.com',
    install_requires = ['numpy','pandas','matplotlib','tdqm', 'torch', 'torchvision', 'sklearn', 'timm', 'visualpriors'],
    url = 'https://github.com/ColinConwell/DeepDive',
    include_package_data=True,
)
