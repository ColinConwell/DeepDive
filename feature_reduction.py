import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, time, pickle, argparse
sys.path.append('..')

import torch as torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA

from feature_extraction import *
from model_options import *

def get_prepped_model(model_string):
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    model = eval(model_call)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        
    return(model)

def check_model(model_string, model = None):
    model_options = get_model_options()
    if model_string not in model_options and model == None:
        raise ValueError('model_string not available in prepped models. Please supply model object.')

def check_reduction_inputs(feature_maps = None, model_inputs = None):
    if feature_maps == None and model_inputs == None:
        raise ValueError('Neither feature_maps nor model_inputs are defined.')
        
    if model_inputs is not None and not isinstance(model_inputs, (DataLoader, torch.Tensor)):
        raise ValueError('model_inputs not supplied in recognizable format.')

def get_feature_map_filepaths(model_string, feature_map_names, output_dir):
    return {feature_map_name: os.path.join(output_dir, feature_map_name + '.npy')
                                for feature_map_name in feature_map_names}

def torch_corrcoef(m):
    #calculate the covariance matrix
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov_m = 1 / (x.size(1) - 1) * x.mm(x.t())
    
    #convert covariance to correlation
    d = torch.diag(cov_m)
    sigma = torch.pow(d, 0.5)
    cor_m = cov_m.div(sigma.expand_as(cov_m))
    cor_m = cor_m.div(sigma.expand_as(cor_m).t())
    cor_m = torch.clamp(cor_m, -1.0, 1.0)
    return cor_m

def srp_extraction(model_string, model = None, feature_maps = None, model_inputs=None, output_dir='./srp_arrays', 
                   n_projections=None, eps=0.1, seed = 0, keep_feature_maps = True):
    
    check_model(model_string, model)
    check_reduction_inputs(feature_maps, model_inputs)
        
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    if n_projections is None:
        if feature_maps is None:
            if isinstance(model_inputs, torch.Tensor):
                n_samples = len(model_inputs)
            if isinstance(model_inputs, DataLoader):
                n_samples = len(model_inputs.dataset)
        if feature_maps is not None:
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    print('Computing {} SRPs for {} on {}...'.format(n_projections, model_string, device_name))

    output_dir = os.path.join(output_dir, str(n_projections) + '_' + str(seed), model_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if feature_maps is None:
        if model == None:
            model = get_prepped_model(model_string)
        
        model = prep_model_for_extraction(model)
        
        feature_map_names = get_empty_feature_maps(model, names_only = True)
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
        if not all([os.path.exists(file) for file in output_filepaths.values()]):
            feature_maps = get_all_feature_maps(model, model_inputs)
            
    if feature_maps is not None:
        feature_map_names = list(feature_maps.keys())
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
    srp_feature_maps = {}
    for feature_map_name in tqdm(feature_map_names):
        output_filepath = output_filepaths[feature_map_name]
        if not os.path.exists(output_filepath):
            feature_map = feature_maps[feature_map_name]
            if feature_map.shape[1] >= n_projections:
                srp = SparseRandomProjection(n_projections, random_state=seed)
                srp_feature_maps[feature_map_name] = srp.fit_transform(feature_map) 
            if feature_map.shape[1] <= n_projections:
                srp_feature_maps[feature_map_name] = feature_map
            np.save(output_filepath, srp_feature_maps[feature_map_name])
        if os.path.exists(output_filepath):
            srp_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
        if not keep_feature_maps:
                feature_maps.pop(feature_map_name)
            
    return(srp_feature_maps)

def rdm_extraction(model_string, model = None, feature_maps = None, model_inputs = None, output_dir='./rdm_arrays', 
                   use_torch_corr = False, append_suffix = False, verbose=False):
    
    check_model(model_string, model)
    check_reduction_inputs(feature_maps, model_inputs)
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    if verbose:
        print('Computing RDMS for {} on {}...'.format(model_string, device_name))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, model_string + '_rdms.pkl' 
                           if append_suffix else model_string + '.pkl')
    
    if os.path.exists(output_file):
        model_rdms = pickle.load(open(output_file,'rb'))
        
    if not os.path.exists(output_file):
        if feature_maps is None:
            if model == None:
                model = get_prepped_model(model_string)
             
            model = prep_model_for_extraction(model)
            feature_maps = get_all_feature_maps(model, model_inputs, numpy = not use_torch_corr)
        
        model_rdms = {}
        for model_layer in tqdm(feature_maps, leave=False):
            if use_torch_corr:
                feature_map = feature_maps[model_layer]
                if torch.cuda.is_available():
                    feature_map = feature_map.cuda()
                model_rdm = torch_corrcoef(feature_map).cpu()
                model_rdms[model_layer] = model_rdm.numpy()
            if not use_torch_corr:
                model_rdms[model_layer] = np.corrcoef(feature_maps[model_layer])
        with open(output_file, 'wb') as file:
            pickle.dump(model_rdms, file)
            
    return(model_rdms)

def pca_extraction(model_string, model = None, feature_maps = None, model_inputs=None, output_dir='./pca_arrays', 
                   n_components=None, use_imagenet_pca = True, imagenet_sample_path = None, keep_feature_maps = True):
    
    check_model(model_string, model)
    check_reduction_inputs(feature_maps, model_inputs)
    
    if use_imagenet_pca == True and imagenet_sample_path is None:
        raise ValueError('use_imagenet_pca selected, but imagenet_sample_path not specified.')
    
    if feature_maps is None:
        if isinstance(model_inputs, torch.Tensor):
            n_samples = len(model_inputs)
        if isinstance(model_inputs, DataLoader):
            n_samples = len(model_inputs.dataset)
    if feature_maps is not None:
        n_samples = next(iter(feature_maps.values())).shape[0]
    
    if n_components is not None:
        if n_components > 1000 and use_imagenet_pca:
            raise ValueError('Requesting more components than are available with PCs from imagenet sample.')
        if n_components > n_samples: 
            raise ValueError('Requesting more components than are available with stimulus set sample size.')
            
    if n_components is None:
        if use_imagenet_pca:
            n_components = 1000
        if not use_imagenet_pca:
            n_components = n_samples
        
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    pca_type = 'imagenet_1000' if use_imagenet_pca else 'stimulus_direct'
    pca_printout = '1000 ImageNet PCs' if use_imagenet_pca else 'up to {} Stimulus PCs'.format(n_components) 
    
    print('Computing {} for {} on {}...'.format(pca_printout, model_string, device_name))

    output_dir = os.path.join(output_dir, pca_type, model_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model == None:
        model = get_prepped_model(model_string)
        
    model = prep_model_for_extraction(model)
        
    if feature_maps is None:
        feature_map_names = get_empty_feature_maps(model, names_only = True)
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
        if not all([os.path.exists(file) for file in output_filepaths.values()]):
            print('Now extracting feature maps for stimulus set...')
            feature_maps = get_all_feature_maps(model, model_inputs)
            
    if feature_maps is not None:
        feature_map_names = list(feature_maps.keys())
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
    if not all([os.path.exists(file) for file in output_filepaths.values()]) and use_imagenet_pca:
        imagenet_images, imagenet_transforms = np.load(imagenet_sample_path), get_image_transforms()['imagenet']
        imagenet_loader = DataLoader(Array2DataSet(imagenet_images, imagenet_transforms), batch_size=64)

        print('Now extracting feature maps for imagenet_sample...')
        imagenet_feature_maps = get_all_feature_maps(model, imagenet_loader)
    
    print('Computing PCA transforms...')
    pca_feature_maps = {}
    for feature_map_name in tqdm(feature_map_names):
        output_filepath = output_filepaths[feature_map_name]
        if not os.path.exists(output_filepath):
            feature_map = feature_maps[feature_map_name]
            n_features = feature_map.shape[1]
            if n_components > n_features:
                n_components = n_features
            if use_imagenet_pca:
                imagenet_feature_map = imagenet_feature_maps[feature_map_name]
                pca = PCA(n_components, random_state=0).fit(imagenet_feature_map)
                pca_feature_maps[feature_map_name] = pca.transform(feature_map)
            if not use_imagenet_pca:
                pca = PCA(n_components, random_state=0).fit(feature_map)
                pca_feature_maps[feature_map_name] = pca.transform(feature_map)
            np.save(output_filepath, pca_feature_maps[feature_map_name])
        if os.path.exists(output_filepath):
            pca_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
        if not keep_feature_maps:
            feature_maps.pop(feature_map_name)
            if use_imagenet_pca:
                imagenet_feature_maps.pop(feature_map_name)
            
    return(pca_feature_maps)

