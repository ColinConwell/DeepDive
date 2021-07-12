import os, sys, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

sys.path.append('../model_options')
from feature_extraction import *
from model_options import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_metadata(model_option, convert_to_dataframe = True):
    model_name = model_option['model_name']
    train_type = model_option['train_type']
    model = eval(model_option['call'])
    model = prep_model_for_extraction(model)
    
    layer_metadata = get_feature_map_metadata(model)
    layer_count = len(layer_metadata.keys())
    layer_order = [model_layer for model_layer in layer_metadata]
    feature_counts = [layer_metadata[layer]['feature_count'] 
                      for layer in layer_metadata]
    parameter_counts = [layer_metadata[layer]['parameter_count'] 
                        for layer in layer_metadata]
    feature_map_shapes = [layer_metadata[layer]['feature_map_shape'] 
                          for layer in layer_metadata]
    total_feature_count = int(np.array(feature_counts).sum())
    total_parameter_count = int(np.array(parameter_counts).sum())
    model_metadata = {'total_feature_count': total_feature_count,
                      'total_parameter_count': total_parameter_count,
                      'layer_count': layer_count,
                      'layer_metadata': layer_metadata}
    
    if not convert_to_dataframe:
        return(model_metadata)
        
    if convert_to_dataframe:

        model_metadata_dictlist = []
        
        for layer_index, layer in enumerate(layer_metadata):
            model_metadata_dictlist.append({'model': model_name, 'train_type': train_type,
                                            'model_layer': layer, 'model_layer_index': layer_index + 1,
                                            'model_layer_depth': (layer_index + 1) / layer_count,
                                            'feature_count': layer_metadata[layer]['feature_count'],
                                            'parameter_count': layer_metadata[layer]['parameter_count']})

        return(pd.DataFrame(model_metadata_dictlist))

