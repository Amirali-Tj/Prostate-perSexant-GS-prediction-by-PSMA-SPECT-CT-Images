import torch
import torchvision.models as models
import numpy as np
from pprint import pprint



resnet = models.resnet101(pretrained=True)
torch_weights = np.array(resnet.conv1.weight.data) # channel first
torch_bias    = np.array(resnet.conv1.bias) 

Conv1W = np.transpose(
    torch_weights ,
    axes = (2 , 3 , 1 , 0)
)

channel_last_weights = {
    "conv1" : Conv1W
}

# some are bias true and some are bias false
#layers = dict(resnet.named_children())
#print(layers["layer1"][0]) 
# next ....