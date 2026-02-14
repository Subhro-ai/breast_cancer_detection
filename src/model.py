import torch.nn as nn
import torchvision.models as models

def get_model():

    model = models.densenet121(pretrained=True)

 
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)

    return model
