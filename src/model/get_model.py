'''
set simple cnn 
'''
from .resnet import resnet_simplest, resnet_simple, resnet34
from dataclasses import dataclass

def get_model(model):
    models = {
    'resnet_simplest' : resnet_simplest,
    'resnet_simple': resnet_simple,
    'resnet34': resnet34,
    }
    return models.get(model)
    

