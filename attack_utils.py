import numpy as np

def create_layer_labels(sample_names, layer: int):

    if layer > len(sample_names[0]) or layer < 1:
        raise ValueError(f"Layer cannot exceed 4 or be less than 1")
    
    layer_labels = []

    for sample in sample_names:
        if list(sample)[layer-1] == 'C':
            layer_labels.append(0)
        elif list(sample)[layer-1] == 'M':
            layer_labels.append(1)
        else:
            layer_labels.append(2)

    return np.array(layer_labels)
