from .model_opts.feature_extraction import *
from .model_opts.feature_reduction import *
from .model_opts.model_options import *
from .model_opts.mapping_methods import *
from .model_opts.model_metadata import *
from .model_opts.model_options import *
from .utils import *

def get_prepped_model(model_string):
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    model = eval(model_call)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return(model)

def get_all_feature_maps(model, inputs, layers_to_retain=None, remove_duplicates=True,
                         include_input_space = False, flatten=True, numpy=True, use_tqdm = True):

    check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)

    if isinstance(inputs, DataLoader):
        input_size, dataset_size, start_index = inputs.dataset[0].shape, len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model, next(iter(inputs))[:3], input_size,
                                              dataset_size, layers_to_retain, remove_duplicates)

        if include_input_space:
            input_map = {'Input': torch.empty(dataset_size, *input_size)}
            feature_maps = {**input_map, **feature_maps}


        for imgs in tqdm(inputs, desc = 'Feature Extraction (Batch)') if use_tqdm else inputs:
            imgs = imgs.cuda() if next(model.parameters()).is_cuda else imgs
            batch_feature_maps = get_feature_maps(model, imgs, layers_to_retain, remove_duplicates = False)

            if include_input_space:
                batch_feature_maps['Input'] = imgs.cpu()

            for map_i, map_key in enumerate(feature_maps):
                feature_maps[map_key][start_index:start_index+imgs.shape[0],...] = batch_feature_maps[map_key]
            start_index += imgs.shape[0]

    if not isinstance(inputs, DataLoader):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)

        if include_input_space:
            feature_maps = {**{'Input': inputs.cpu()}, **feature_maps}

    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)

    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)

    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy()

    return feature_maps
