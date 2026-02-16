import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.efficientnet_b3(pretrained=True)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)

    return model
