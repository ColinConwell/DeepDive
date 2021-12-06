# DeepDive (into Deep Neural Networks)

Designed for deep net feature extraction, dimensionality reduction, and benchmarking, this repo contains a number of convenience functions for loading and instrumentalizing a variety of (PyTorch) models. Models available include those from:

- the [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models) (Timm) library
- the [Torchvision](https://pytorch.org/vision/stable/models.html) model zoo
- the [Taskonomy](http://taskonomy.stanford.edu/) (visual_priors) project
- the [VISSL](https://vissl.ai/) (self-supervised) model zoo
- the [Detectron2](https://github.com/facebookresearch/detectron2) model zoo
- ISL's [MiDas](https://github.com/isl-org/MiDaS) models, FaceBook's [DINO](https://github.com/facebookresearch/dino) models...

And many more. This repository is a work in progress; please cite any issues you encounter.

### Installation:

```
pip install git+https://github.com/ColinConwell/DeepDive.git
```

If you find this repository useful, please consider citing it with the following BibTex:

```BibTeX
@misc{conwell2021deepdive,
  author =       {Colin Conwell},
  title =        {DeepDive},
  howpublished = {\url{https://github.com/ColinConwell/DeepDive}},
  year =         {2021}
}
```

(Also remember to cite any of the specific models you use by referring to their original sources linked in the model_typology.csv file).
