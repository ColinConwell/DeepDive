import pandas as pd
import os, sys, torch, timm

import torchvision.models as models

path_dir = os.path.dirname(os.path.abspath(__file__))
model_typology = pd.read_csv(path_dir + '/model_typology.csv')
model_typology['model_name'] = model_typology['model']
model_typology['model_type'] = model_typology['model_type'].str.lower()

def define_torchvision_options():
    torchvision_options = {}
    
    model_types = ['imagenet','inception','segmentation', 'detection', 'video']
    pytorch_dirs = dict(zip(model_types, ['.','.','.segmentation.', '.detection.', '.video.']))

    torchvision_typology = model_typology[model_typology['model_source'] == 'torchvision'].copy()
    torchvision_typology['model_type'] = torchvision_typology['model_type'].str.lower()
    training_calls = {'random': '(pretrained=False)', 'pretrained': '(pretrained=True)'}
    for index, row in torchvision_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        model_source = 'torchvision'
        for training in ['random', 'pretrained']:
            train_type = row['model_type'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            model_call = 'models' + pytorch_dirs[model_type] + model_name + training_calls[training]
            torchvision_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                                  'train_type': train_type, 'model_source': model_source, 'call': model_call})
            
    return torchvision_options

from visual_priors import taskonomy_network

def instantiate_taskonomy_model(model_name, verbose = False):
    weights = torch.load(path_dir + '/task_weights/{}_encoder.pth'.format(model_name))
    if verbose: print('{} weights loaded succesfully.'.format(model_name))
    model = taskonomy_network.TaskonomyEncoder()
    model.load_state_dict(weights['state_dict'])
    
    return model

def define_taskonomy_options():
    taskonomy_options = {}

    task_typology = model_typology[model_typology['train_type'] == 'taskonomy'].copy()
    for index, row in task_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type, model_source = 'taskonomy', 'taskonomy'
        model_string = model_name + '_' + train_type
        model_call = "instantiate_taskonomy_model('{}')".format(model_name)
        taskonomy_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                            'train_type': train_type, 'model_source': model_source, 'call': model_call})
        
    taskonomy_options['random_weights_taskonomy'] = ({'model_name': 'random_weights', 'model_type': 'taskonomy',
                                                      'train_type': 'taskonomy', 'model_source': 'taskonomy',
                                                      'call': 'taskonomy_network.TaskonomyEncoder()'})
            
    return taskonomy_options

def define_timm_options():
    timm_options = {}

    timm_typology = model_typology[model_typology['model_source'] == 'timm'].copy()
    for index, row in timm_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        model_source = 'timm'
        for training in ['random', 'pretrained']:
            train_type = row['model_type'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            train_bool = False if training == 'random' else True
            model_call = "timm.create_model('{}', pretrained = {})".format(model_name, train_bool)
            timm_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                           'train_type': train_type, 'model_source': model_source, 'call': model_call})
            
    return timm_options

def get_model_options(model_type = None, train_type=None, model_source=None):
    model_options = {**define_torchvision_options(), **define_taskonomy_options(), **define_timm_options()}
    
    if model_type is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['model_type'] in model_type}
        
    if train_type is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['train_type'] in train_type}
        
    if model_source is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['model_source'] in model_source}
        
    return model_options
    
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def taskonomy_transform(image):
    return (functional.to_tensor(functional.resize(image, (256,256))) * 2 - 1)#.unsqueeze_(0)

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std':  [0.229, 0.224, 0.225]}

torchvision_transforms = {
    'random': [transforms.Resize((224,224)), transforms.ToTensor()],
    'imagenet': [transforms.Resize((224,224)), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])],
    'inception': [transforms.Resize((299,299)), transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])],
    'detection': [transforms.Resize((224,224)), transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])],
    'segmentation': 'https://github.com/pytorch/vision/blob/master/references/segmentation/train.py',
    'video': 'https://github.com/pytorch/vision/blob/master/references/video_classification/train.py',
}

transform_options = {**torchvision_transforms, 'taskonomy': taskonomy_transform}

def get_transform_options():
    return transform_options
    
def get_recommended_transforms(model_query, input_type = 'PIL'):
    model_types = model_typology['model_type'].unique()
    if model_query in get_model_options():
        model_type = get_model_options()[model_query]['model_type']
    if model_query in model_types:
        model_type = model_query
    if model_query not in list(get_model_options()) + list(model_types):
        raise ValueError('Query is neither a model_string nor a model_type.')
    composable = ['imagenet', 'inception','detection']
    reference = ['segmentation', 'video']
    functionals = ['taskonomy']
    
    if model_type in composable:
        if input_type == 'PIL':
            recommended_transforms = transform_options[model_type]
        if input_type == 'numpy':
            recommended_transforms = [transforms.ToPILImage()] + transform_options[model_type]
        return transforms.Compose(recommended_transforms)
    
    if model_type in functionals:
        if input_type == 'PIL':
            return transform_options[model_type]
        if input_type == 'numpy':
            def functional_from_numpy(image):
                image = functional.to_pil_image(image)
                return transform_options[model_type](image)
            return functional_from_numpy
        
    if model_type in reference:
        recommended_transforms = transform_options[model_type]
        print('Please see {} for best practices.'.format(transform_options))
        
    if model_type not in transform_options:
        print('No reference available for this model class.')
    
training_printouts = {
    'random': 'randomly initialized',
    'imagenet': 'pretrained on imagenet',
    'taskonomy': 'pretrained on taskonomy',
}
    

def get_training_printouts(train_type = None):
    if train_type is None:
        return training_printouts
    if train_type is not None:
        return training_printouts[train_type]
