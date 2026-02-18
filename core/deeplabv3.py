import torchvision.models as models
import torch.nn as nn

def get_model(num_classes):
    model=models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4]=nn.Conv2d(256,num_classes,1)
    return model