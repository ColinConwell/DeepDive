# DeepDive (into Deep Neural Networks)

Designed for deep net feature extraction, dimensionality reduction, and benchmarking, this repo contains a number of convenience functions for loading and instrumentalizing a variety of (PyTorch) models. Models available include those from:

- the [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models) (Timm) library
- the [Torchvision](https://pytorch.org/vision/stable/models.html) model zoo
- the [Taskonomy](http://taskonomy.stanford.edu/) (visual_priors) project
- the [VISSL](https://vissl.ai/) (self-supervised) model zoo
- the [Detectron2](https://github.com/facebookresearch/detectron2) model zoo
- ISL's [MiDas](https://github.com/isl-org/MiDaS) models, FaceBook's [DINO](https://github.com/facebookresearch/dino) models...

Check out these repos that benchmark these models on [human fMRI](https://github.com/ColinConwell/DeepNSD) and [mouse optical physiology](https://github.com/ColinConwell/DeepMouseTrap) data.

A tutorial that demonstrates the main functionality of this pipeline in both behavior and brains may be found [here](https://colab.research.google.com/drive/1CvOpeKL4xRDbHkpPXGlSDs-JyD-438vl#scrollTo=Jd9vyENcvsIg).

This repository is a work in progress; please feel free to file any issues you find.

If you find this repository useful, please consider citing the work that fueled its most recent version:

 ```bibtex
@article{conwell2023pressures,
  title={What can 1.8 billion regressions tell us about the pressures shaping high-level visual representation in brains and machines},
  author={Conwell, Colin and Prince, Jacob S and Kay, Kendrick N and Alvarez, George A and Konkle, Talia},
  journal={bioRxiv},
  year={2023}
}
 ```

(Also remember to cite any of the specific models you use by referring to their original sources linked in the model_typology.csv file).

## 2024 Update: *DeepDive* to *DeepJuice*

+ **Squeezing your deep nets for science!**

Since the release of ChatGPT, our team has been working on a new, highly-accelerated version of this codebase called **Deepjuice** -- effectively, a bottom-up reimplementation of all DeepDive functionalities that allows for end-to-end benchmarking (feature extraction, SRP, PCA, CKA, RSA, and regression) without ever removing data from the GPU. 

**DeepJuice** is currently in private beta, but if you're interested in trying out, please feel free to contact me (Colin Conwell) by email: conwell[at]g[dot]harvard[dot]edu)
