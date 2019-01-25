import numpy as np

def get_number_of_parameters(model):
    return sum([np.prod(list(p.shape)) for p in model.parameters()])
